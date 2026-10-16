"""Concurrent multi-match supervisor — the engine behind Milestone 4C's
``python -m agent``.

Non-blocking parent loop: repeatedly discovers this agent's active matches
via ``client.sessions()``, and keeps one independent worker *process*
running per active ``session_id`` (see ``altruagent.worker`` for the child
side, and why processes rather than threads/asyncio — module-level
contestant state and crash isolation both need a real OS-process boundary,
not just a Python-level one).

The parent never plays a match itself. It only starts/tracks/reaps worker
processes and owns the one thing that's meaningless at the per-worker
level: failed-match cooldown. A worker is one-shot (one match, then it
exits) — cooldown is inherently a "should I start a *new* one for this
session_id soon after the last one failed" decision, which only the
long-lived supervisor can make.

Explicitly uses the ``spawn`` multiprocessing context everywhere (never the
platform default) so behavior is identical and Windows-compatible
regardless of what OS this ever runs on — Windows has no ``fork`` at all.
"""

from __future__ import annotations

import multiprocessing
import os
import time
from collections.abc import Callable, Collection
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from .console import print_line
from .errors import AuthenticationError, PlatformError
from .official import (
    is_fatal_auth_error,
    is_registration_incomplete,
    new_execution_id,
    registration_wait_message,
)
from .runner import TRANSIENT_GIVE_UP_SECONDS
from .worker import (
    EXIT_CONNECTION_LOST,
    EXIT_KEY_REFUSED,
    EXIT_NOT_CONNECTED,
    EXIT_REGISTRATION_INCOMPLETE,
    EXIT_SEAT_BUSY,
    EXIT_SUCCESS,
    TournamentWorkerInput,
    WorkerInput,
    _process_entry,
    _tournament_process_entry,
)

if TYPE_CHECKING:
    from .client import AltruAgentClient
    from .models import OfficialAssignment
    from .official import OfficialAgentClient

DEFAULT_DISCOVERY_INTERVAL_SECONDS = 15.0
DEFAULT_COOLDOWN_SECONDS = 60.0
DEFAULT_SHUTDOWN_JOIN_TIMEOUT_SECONDS = 5.0

_MP_CONTEXT = multiprocessing.get_context("spawn")


def _log(message: str) -> None:
    print(f"[agent] {message}")


class WorkerRegistry:
    """Tracks live worker processes keyed by ``session_id``.

    A thin, deliberately dumb bookkeeping object — no policy lives here
    (cooldown/eligibility decisions stay in ``run_once_concurrent``). Public
    (not underscore-prefixed) so tests can construct and inspect one
    directly without going through a full supervisor tick.
    """

    def __init__(self) -> None:
        self._processes: dict[str, multiprocessing.process.BaseProcess] = {}

    def is_active(self, session_id: str) -> bool:
        return session_id in self._processes

    def pids(self) -> dict[str, int]:
        """Currently-tracked ``{session_id: pid}`` — a read-only view used
        by callers (and the developer smoke test) that want to verify real,
        distinct OS processes are running, not just that the registry
        thinks something is active.
        """
        return {session_id: process.pid for session_id, process in self._processes.items()}

    def start(self, session_id: str, process: multiprocessing.process.BaseProcess) -> None:
        self._processes[session_id] = process

    def reap_finished(self) -> dict[str, int]:
        """Remove every worker whose process has exited since the last
        call, and return ``{session_id: exitcode}`` for each. Calls
        ``.join()`` on each (already-exited) process to promptly reclaim OS
        resources — cheap and instant on a process that's already dead.
        """
        finished: dict[str, int] = {}
        for session_id, process in list(self._processes.items()):
            if not process.is_alive():
                process.join()
                finished[session_id] = process.exitcode
                del self._processes[session_id]
        return finished

    def terminate(self, session_id: str, timeout: float) -> None:
        """Forcibly stop one worker (same semantics as ``terminate_all``)."""
        process = self._processes.pop(session_id, None)
        if process is None:
            return
        if process.is_alive():
            process.terminate()
        process.join(timeout)

    def keys(self) -> list[str]:
        return list(self._processes)

    def terminate_all(self, timeout: float) -> None:
        """Forcibly stop every remaining worker and reclaim it. Uses
        ``.terminate()`` (immediate, no grace period) rather than asking
        nicely — on shutdown we specifically do NOT want a worker to get
        the chance to make one more API call (e.g. a "graceful" resign);
        each worker also independently handles its own ``KeyboardInterrupt``
        (see ``altruagent.worker``), so in the common case this just cleans
        up stragglers that didn't exit fast enough on their own.
        """
        for process in self._processes.values():
            if process.is_alive():
                process.terminate()
        for process in self._processes.values():
            process.join(timeout)
        self._processes.clear()

    def __len__(self) -> int:
        return len(self._processes)


def run_once_concurrent(
    client: AltruAgentClient,
    *,
    registry: WorkerRegistry,
    failed_until: dict,
    agent_id: str,
    now: Callable[[], float] = time.monotonic,
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
    process_factory: Callable[..., multiprocessing.process.BaseProcess] = _MP_CONTEXT.Process,
) -> None:
    """One non-blocking supervisor tick.

    1. Reap any workers that have finished since the last tick, applying a
       cooldown to any that exited with a non-success code.
    2. Discover this agent's current sessions and, for every ``active``
       match that doesn't already have a live worker and isn't in
       cooldown, start one. ``waiting``/``completed`` matches are never
       considered — only ``sessions.active`` is inspected.

    Never blocks on any worker; always returns immediately, regardless of
    how many matches are in flight.
    """
    current_time = now()

    for session_id, exitcode in registry.reap_finished().items():
        if exitcode != EXIT_SUCCESS:
            failed_until[session_id] = current_time + cooldown_seconds
            _log(
                f"worker for session_id={session_id} exited with code {exitcode} — "
                f"cooling down for {cooldown_seconds:.0f}s"
            )
        else:
            failed_until.pop(session_id, None)
            _log(f"worker for session_id={session_id} finished")

    sessions = client.sessions()
    for match in sessions.active:
        if registry.is_active(match.session_id):
            continue
        retry_at = failed_until.get(match.session_id)
        if retry_at is not None and current_time < retry_at:
            continue

        worker_input = WorkerInput(
            session_id=match.session_id,
            tournament_id=match.tournament_id,
            game_type=match.game_type,
            agent_id=agent_id,
        )
        process = process_factory(target=_process_entry, args=(worker_input,), daemon=True)
        process.start()
        registry.start(match.session_id, process)
        _log(
            f"started worker pid={process.pid} for session_id={match.session_id} "
            f"game_type={match.game_type} tournament_id={match.tournament_id}"
        )


def run_forever_concurrent(
    client: AltruAgentClient,
    *,
    agent_id: str,
    discovery_interval: float = DEFAULT_DISCOVERY_INTERVAL_SECONDS,
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
    shutdown_join_timeout: float = DEFAULT_SHUTDOWN_JOIN_TIMEOUT_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
    process_factory: Callable[..., multiprocessing.process.BaseProcess] = _MP_CONTEXT.Process,
    max_iterations: int | None = None,
) -> None:
    """Repeatedly discover this agent's active matches and keep one worker
    process per active ``session_id`` running, until interrupted (or, for
    tests, until ``max_iterations`` ticks have elapsed).

    Whatever ends the loop — ``KeyboardInterrupt``, an unexpected exception
    (e.g. the parent's own discovery call hitting ``AuthenticationError``,
    which is *not* caught here and is treated as fatal — see module
    docstring), or ``max_iterations`` being reached — every still-running
    worker is terminated and joined (bounded by ``shutdown_join_timeout``)
    in a ``finally`` block before anything propagates further. No match is
    ever resigned on shutdown.
    """
    registry = WorkerRegistry()
    failed_until: dict = {}
    ticks = 0
    try:
        while max_iterations is None or ticks < max_iterations:
            run_once_concurrent(
                client,
                registry=registry,
                failed_until=failed_until,
                agent_id=agent_id,
                now=now,
                cooldown_seconds=cooldown_seconds,
                process_factory=process_factory,
            )
            ticks += 1
            sleep(discovery_interval)
    finally:
        if len(registry) > 0:
            _log(f"stopping {len(registry)} active worker(s)...")
            registry.terminate_all(shutdown_join_timeout)


# -- official tournament runtime -------------------------------------------------

DEFAULT_TOURNAMENT_POLL_SECONDS = 10.0
# A seat must be missing from this many consecutive successful assignment
# listings before its still-running worker is stopped — a match that just
# ended can drop off the list a moment before its worker notices.
MISSING_POLLS_BEFORE_STOP = 2
WAITING_MESSAGE = "Waiting for your next game..."
# Said once when a game's worker finds the Official Agent Key refused (it was
# rotated or revoked while this runtime was running). The runtime's own agent
# session still works for up to about an hour, so it would otherwise keep
# picking up games it can't play. From then on it starts no new game; the
# games already running keep playing. A new key needs a restart: .env is read
# once, at startup.
KEY_REFUSED_MESSAGE = (
    "Your Official Agent Key is no longer accepted, so no new game will start. Put your new key in .env "
    "(ALTRUAGENT_OFFICIAL_AGENT_KEY) and restart this process; check too that your agent is still Self-hosted. "
    "Games already running keep playing for now."
)
# After seat_busy, retry the seat once the backend's 30 s execution lease
# could have lapsed (a previous run releasing it, or this runtime's own
# re-authenticated session); a seat another live runtime keeps renewing
# just stays busy.
SEAT_BUSY_RETRY_SECONDS = 35.0
# A seat whose worker keeps failing (a game the server lost answers
# SESSION_NOT_FOUND until the platform closes it, which takes a few minutes;
# an agent that crashes on the same state again) is retried less and less
# often: the cooldown doubles with each failure in a row, up to this. A
# finished match or a seat_busy resets it, and a worker that ran at least this
# long before failing starts a new count.
SEAT_FAILURE_BACKOFF_MAX_SECONDS = 600.0
# A worker that lost the connection to its game (EXIT_CONNECTION_LOST: the
# in-game retries gave up after runner.TRANSIENT_GIVE_UP_SECONDS) after
# playing for at least WORKER_PLAYED_SECONDS is picked up again after
# RESTART_AFTER_PLAY_SECONDS, not a minute or more, and starts a new count.
# The reason decides it, not only the time: an agent that crashes, or a
# worker that never reached its game, keeps the doubling cooldown however
# long it ran. A worker that lost the connection almost at once also does,
# so a game that keeps failing can't cost a new seat grant every ~100 s.
WORKER_PLAYED_SECONDS = 30.0
RESTART_AFTER_PLAY_SECONDS = 10.0
# A worker that never reached its game because of a temporary problem
# (EXIT_NOT_CONNECTED) is replaced after this pause (0: in the same poll)
# while the game's connect window is open, however often it happened: an
# agent that isn't connected by the deadline loses the game as a no-show.
# No hot loop: such a worker has already retried for a minute or more itself
# (worker.FIRST_GRANT_RETRY_SECONDS, runner.TRANSIENT_GIVE_UP_SECONDS).
NOT_CONNECTED_RETRY_SECONDS = 0.0
# The connect window counts as open for at least this long after a game is
# first listed (or until its connect_deadline_at, if later): a computer whose
# clock is off must not cut it short, and retrying past the real deadline
# costs little (the platform then answers that the game has ended).
MIN_CONNECT_WINDOW_SECONDS = 300.0
# While the platform says the event registration is incomplete (for example
# the Official Rules were updated), the runtime keeps trying at this pace and
# repeats what to do at most every REGISTRATION_REMINDER_SECONDS.
REGISTRATION_RETRY_SECONDS = 30.0
REGISTRATION_REMINDER_SECONDS = 300.0
# The agent session lasts about an hour, so the runtime signs in again with
# the Official Agent Key now and then. When that hits a temporary problem
# (429, 5xx, a session the platform couldn't start or didn't accept), the
# runtime keeps its running games and tries again after a pause: 10 s,
# doubling to at most 60 s, and a full minute after "too many attempts".
AUTH_RETRY_BASE_SECONDS = 10.0
AUTH_RETRY_MAX_SECONDS = 60.0
AUTH_RATE_LIMIT_WAIT_SECONDS = 60.0


# The two kinds of game an assignment can be (its ``context``):
# ``python -m agent --match`` plays TESTING games (test matches),
# ``--tournament`` plays TOURNAMENT games, and both flags together play both.
TESTING = "testing"
TOURNAMENT = "tournament"
ALL_KINDS = frozenset({TESTING, TOURNAMENT})
TEST_MATCH_WAITING_NOTE = "Test match waiting: run with --match to play it"


def _utc_now() -> datetime:
    return datetime.now(UTC)


# Names and labels come from the server (opponent names are chosen by other
# contestants), so they're cleaned before reaching the terminal: no control
# or escape characters, and a bounded length.
_MAX_TEXT = 80


def _clean(text: str | None) -> str:
    cleaned = "".join(ch for ch in (text or "") if ch.isprintable()).strip()
    return cleaned if len(cleaned) <= _MAX_TEXT else cleaned[: _MAX_TEXT - 3].rstrip() + "..."


def _parse_utc(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _format_wait(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _kind_label(assignment: OfficialAssignment) -> str:
    context = (assignment.context or "").strip().lower()
    return {"testing": " (Testing)", "tournament": " (tournament)"}.get(context, "")


def _players_note(players: int, player_count: int | None) -> str:
    """How many of a match's players this runtime picked up at once, when more
    than one (self-play, or several players in one test match)."""
    if players <= 1:
        return ""
    if player_count == players:
        return f", self-play: your agent plays all {players} players"
    if isinstance(player_count, int) and player_count > players:
        return f", your agent plays {players} of the {player_count} players"
    return f", your agent plays {players} players"


def describe_assignment(
    assignment: OfficialAssignment, *, now: datetime | None = None, players: int = 1
) -> list[str]:
    """The lines logged when a game is picked up: the game, then (when the
    server sent them) Testing vs tournament, the tournament and round, the
    opponents, and the connect deadline with the time left. Never prints ids
    or tokens. ``players``: how many of the match's players were picked up
    together (one set of lines for all of them).
    """
    game = _clean(assignment.game_type) or "unknown game"
    lines = [
        f"Match assigned: {game}{_kind_label(assignment)}{_players_note(players, assignment.seat_count)}"
    ]

    name, round_label = _clean(assignment.tournament_name), _clean(assignment.round_label)
    if name and round_label:
        lines.append(f"  Tournament: {name}, {round_label}")
    elif name:
        lines.append(f"  Tournament: {name}")
    elif round_label:
        lines.append(f"  Round: {round_label}")

    opponents = [cleaned for cleaned in (_clean(o) for o in assignment.opponents) if cleaned]
    if opponents:
        lines.append(f"  Opponents: {', '.join(opponents)}")

    if assignment.connect_deadline_at:
        deadline = _parse_utc(assignment.connect_deadline_at)
        if deadline is None:
            lines.append(f"  Connect by: {_clean(assignment.connect_deadline_at)}")
        else:
            at = deadline.strftime("%Y-%m-%d %H:%M:%S UTC")
            left = (deadline - (now or _utc_now())).total_seconds()
            when = f"{_format_wait(left)} left" if left > 0 else "deadline passed"
            lines.append(f"  Connect by: {at} ({when})")
    return lines


def assignment_kind(assignment: OfficialAssignment) -> str:
    """``TESTING`` for a test match, ``TOURNAMENT`` for anything else.

    An assignment without a ``context`` comes from an older backend, which
    hands out only tournament games, so it is a tournament game. So is a
    context this SDK doesn't know: a missed tournament game counts as a loss,
    a missed test match doesn't.
    """
    context = (assignment.context or "").strip().lower()
    return TESTING if context == TESTING else TOURNAMENT


def tournament_waiting_warning(assignment: OfficialAssignment) -> str:
    """The warning a ``--match``-only runtime prints (once per game) for a
    tournament game it won't play. Names the tournament and round when the
    server sent them, else the game."""
    name, round_label = _clean(assignment.tournament_name), _clean(assignment.round_label)
    where = ", ".join(part for part in (name, round_label) if part) or _clean(assignment.game_type)
    detail = f" ({where})" if where else ""
    return (
        f"You have a tournament game waiting{detail}: run with --tournament to play it — "
        "it counts as a loss if your agent doesn't connect within the window."
    )


class TournamentState:
    """Everything the tournament loop carries between ticks — in memory only.

    ``execution_id`` is generated once per runtime (one ``python -m agent
    --tournament``/``--match`` process) and handed to every worker it
    starts: it's how the backend tells this runtime's seat leases apart from
    another runtime's. Never persisted or logged.
    """

    def __init__(self) -> None:
        self.execution_id = new_execution_id()
        self.registry = WorkerRegistry()
        self.failed_until: dict[str, float] = {}
        # Failed workers in a row, per seat (see SEAT_FAILURE_BACKOFF_MAX_SECONDS).
        self.seat_failures: dict[str, int] = {}
        # When (on the ``now`` clock) each seat's current worker started, and
        # until when each listed seat's connect window is open (with the
        # match and deadline it was worked out for: a new game for the same
        # seat gets a new window).
        self.started_at: dict[str, float] = {}
        self.connect_by: dict[str, tuple[tuple[str, str | None], float]] = {}
        # When the registration message was last printed (``now`` clock).
        self.registration_noted_at: float | None = None
        self.missing_polls: dict[str, int] = {}
        self.discovery_failing = False
        # Games (match ids) of a kind this runtime doesn't play that it has
        # already printed a warning or note about: once per game.
        self.noted_games: set[str] = set()
        # Temporary sign-in failures in a row, and when (on the ``now``
        # clock) the next assignment listing may try again.
        self.auth_failures = 0
        self.discovery_retry_at: float | None = None
        # The assignment each seat's worker was started for, while the seat is
        # listed or has a worker: which match (and game) a worker belongs to.
        self.seat_games: dict[str, OfficialAssignment] = {}
        # Seats whose last worker stopped before its game ended (an error, a
        # lost connection, no way in yet, an incomplete registration) and
        # that have no worker now. If the match ends before a new worker
        # finishes it, the runtime says so, once per match.
        self.stopped: dict[str, OfficialAssignment] = {}
        # True once a worker found the Official Agent Key refused
        # (EXIT_KEY_REFUSED): no new worker is started after that.
        self.key_refused = False


def _auth_retry_delay(exc: AuthenticationError, failures: int) -> float:
    if exc.status_code == 429:
        return AUTH_RATE_LIMIT_WAIT_SECONDS
    return min(AUTH_RETRY_MAX_SECONDS, AUTH_RETRY_BASE_SECONDS * 2**failures)


def _connect_by(assignment: OfficialAssignment, current_time: float, wall: datetime) -> float:
    """The end of the assignment's connect window on the ``now`` clock:
    ``MIN_CONNECT_WINDOW_SECONDS`` from now, or its ``connect_deadline_at``
    if that is later."""
    deadline = (
        _parse_utc(assignment.connect_deadline_at) if assignment.connect_deadline_at else None
    )
    left = (deadline - wall).total_seconds() if deadline is not None else 0.0
    return current_time + max(MIN_CONNECT_WINDOW_SECONDS, left)


def _note_registration(
    state: TournamentState, current_time: float, message: str, log: Callable[[str], None]
) -> None:
    if (
        state.registration_noted_at is None
        or current_time - state.registration_noted_at >= REGISTRATION_REMINDER_SECONDS
    ):
        state.registration_noted_at = current_time
        log(message)


def _note_other_kind(
    assignment: OfficialAssignment, state: TournamentState, log: Callable[[str], None]
) -> None:
    """Say once per game that a game of the other kind is waiting: a clear
    warning for a tournament game (missing it is a loss), one quiet line for
    a test match."""
    game = assignment.match_id or assignment.seat_id
    if game in state.noted_games:
        return
    state.noted_games.add(game)
    if assignment_kind(assignment) == TOURNAMENT:
        log(tournament_waiting_warning(assignment))
    else:
        log(TEST_MATCH_WAITING_NOTE)


ONLY_GAMES_ENV = "ALTRUAGENT_ONLY_GAMES"


def _only_games() -> frozenset[str]:
    """Game types this runtime may play (``ALTRUAGENT_ONLY_GAMES=a,b``); empty means every game."""
    raw = os.environ.get(ONLY_GAMES_ENV, "")
    return frozenset(g.strip().lower() for g in raw.split(",") if g.strip())


def _note_other_game(
    assignment: OfficialAssignment, state: TournamentState, log: Callable[[str], None]
) -> None:
    """Say once per game that a game this runtime was told not to play is waiting."""
    game = assignment.match_id or assignment.seat_id
    if game in state.noted_games:
        return
    state.noted_games.add(game)
    log(
        f"Skipping a {_clean(assignment.game_type) or 'game'}{_kind_label(assignment)}: this runtime "
        f"only plays {', '.join(sorted(_only_games()))} ({ONLY_GAMES_ENV})."
    )


def run_tournament_once(
    official: OfficialAgentClient,
    state: TournamentState,
    *,
    agent_spec: str,
    kinds: Collection[str] = ALL_KINDS,
    now: Callable[[], float] = time.monotonic,
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
    shutdown_join_timeout: float = DEFAULT_SHUTDOWN_JOIN_TIMEOUT_SECONDS,
    process_factory: Callable[..., multiprocessing.process.BaseProcess] = _MP_CONTEXT.Process,
    log: Callable[[str], None] = print_line,
    wall_now: Callable[[], datetime] = _utc_now,
) -> None:
    """One non-blocking tournament tick, keyed by ``seat_id`` throughout:

    1. Reap finished workers (a failure puts that seat in cooldown; it is
       retried later, which re-requests its grant — the reconnect path).
       Each failure in a row doubles that seat's cooldown, up to
       ``SEAT_FAILURE_BACKOFF_MAX_SECONDS`` (a worker that ran that long
       first starts a new count). A worker that lost the connection to its
       game (``EXIT_CONNECTION_LOST``) after playing for
       ``WORKER_PLAYED_SECONDS`` starts a new count and is replaced after
       ``RESTART_AFTER_PLAY_SECONDS``. A worker that never reached its game
       because of a temporary problem (``EXIT_NOT_CONNECTED``) is replaced
       after ``NOT_CONNECTED_RETRY_SECONDS`` while the connect window is open.
       One turned away by an incomplete registration is retried every
       ``REGISTRATION_RETRY_SECONDS``. One that found the Official Agent Key
       refused (``EXIT_KEY_REFUSED``) makes the runtime say
       ``KEY_REFUSED_MESSAGE`` once and start no new worker from then on (the
       running ones keep playing). A worker that stopped with an error is
       said to have stopped while the match continues, never that the match
       ended; ``WAITING_MESSAGE`` follows only exits after which nothing of
       that game is pending here. A match whose players this runtime plays
       (self-play) gets one "Match finished." when the last of them is done.
    2. List this agent's active assignments. A transient control-plane
       failure is logged and the tick skipped (the runtime keeps waiting).
       So is a temporary failure to sign in again (``is_fatal_auth_error``
       is false), which also pauses listing for a backoff; running workers
       are never touched. An incomplete registration (for example updated
       Official Rules) is not fatal either: the runtime says what to do and
       keeps trying every ``REGISTRATION_RETRY_SECONDS``. Only a refusal of
       the key itself propagates — it would affect every seat.
    3. A seat whose worker had stopped (``TournamentState.stopped``) and that
       has left the list: its match ended without that worker; said once per
       match. Then start one worker per listed seat of a kind in ``kinds``
       (``assignment_kind``: ``TESTING`` and/or ``TOURNAMENT``) that has none
       and isn't cooling down, logging what was picked up
       (``describe_assignment``, once per match however many of its players
       start). A game of another kind is not played; it gets one warning or
       note per game (``_note_other_kind``).
    4. Stop workers whose seat has left the list for
       ``MISSING_POLLS_BEFORE_STOP`` consecutive listings. Every listed seat
       counts here, whatever its kind: ``kinds`` only decides which games
       are started, never stops one already being played.
    """
    current_time = now()
    # Several of a match's players can be played by this runtime (self-play):
    # what is said about one of them is said once per match and tick.
    said: set[tuple[str, str]] = set()

    def match_of(seat_id: str) -> str:
        assignment = state.seat_games.get(seat_id)
        return (assignment.match_id if assignment is not None else "") or seat_id

    def say(seat_id: str, line: str) -> None:
        if (match_of(seat_id), line) not in said:
            said.add((match_of(seat_id), line))
            log(line)

    def say_waiting() -> None:  # at most once per tick; never after a refused key
        if ("", WAITING_MESSAGE) not in said and not state.key_refused:
            said.add(("", WAITING_MESSAGE))
            log(WAITING_MESSAGE)

    reaped = state.registry.reap_finished()
    finished: list[str] = []  # matches with a player that finished, in order
    idle = False  # whether "Waiting for your next game..." fits these exits
    for seat_id, exitcode in reaped.items():
        state.missing_polls.pop(seat_id, None)
        started = state.started_at.pop(seat_id, None)
        ran = current_time - started if started is not None else 0.0
        # Either way the seat waits out a cooldown before any new worker: a
        # finished seat can linger in the listing for a moment, and a failed
        # one is retried (re-granted) only if it's still assigned afterwards.
        state.failed_until[seat_id] = current_time + cooldown_seconds
        connect_by = state.connect_by[seat_id][1] if seat_id in state.connect_by else None
        if exitcode not in (EXIT_SUCCESS, EXIT_SEAT_BUSY) and seat_id in state.seat_games:
            state.stopped[seat_id] = state.seat_games[seat_id]
        if exitcode == EXIT_SUCCESS:
            state.seat_failures.pop(seat_id, None)
            if match_of(seat_id) not in finished:
                finished.append(match_of(seat_id))
            idle = True
        elif exitcode == EXIT_SEAT_BUSY:
            state.seat_failures.pop(seat_id, None)
            state.failed_until[seat_id] = current_time + SEAT_BUSY_RETRY_SECONDS
            say(
                seat_id,
                f"Another runtime is playing this match with your Official Agent Key; checking again in "
                f"{SEAT_BUSY_RETRY_SECONDS:.0f}s.",
            )
            idle = True
        elif exitcode == EXIT_KEY_REFUSED:
            # No retry can help until the contestant restarts with the new key.
            if not state.key_refused:
                state.key_refused = True
                log(KEY_REFUSED_MESSAGE)
        elif exitcode == EXIT_REGISTRATION_INCOMPLETE:
            # The worker printed what to do; nothing is counted against the game.
            state.failed_until[seat_id] = current_time + REGISTRATION_RETRY_SECONDS
            say(
                seat_id,
                f"That game can't start until your registration is complete; checking again in "
                f"{REGISTRATION_RETRY_SECONDS:.0f}s.",
            )
            idle = True
        elif (
            exitcode == EXIT_NOT_CONNECTED and connect_by is not None and current_time < connect_by
        ):
            # Nothing was played yet and the connect window is open: try again
            # right away, however often it happened, so a hiccup isn't a no-show.
            state.failed_until[seat_id] = current_time + NOT_CONNECTED_RETRY_SECONDS
            say(seat_id, "Couldn't reach that game yet (a temporary problem); trying again now.")
        else:
            if (
                exitcode == EXIT_CONNECTION_LOST
                and ran >= TRANSIENT_GIVE_UP_SECONDS + WORKER_PLAYED_SECONDS
            ):
                failures = 1  # it was playing: a new failure, not one more in a row
                wait = RESTART_AFTER_PLAY_SECONDS
            else:
                # An agent error, a failure before the game was reached (the
                # connect window has closed), or a connection lost almost at
                # once: one more in a row, unless the worker ran a long while.
                long_run = (
                    exitcode != EXIT_NOT_CONNECTED and ran >= SEAT_FAILURE_BACKOFF_MAX_SECONDS
                )
                failures = 1 if long_run else state.seat_failures.get(seat_id, 0) + 1
                wait = min(
                    max(SEAT_FAILURE_BACKOFF_MAX_SECONDS, cooldown_seconds),
                    cooldown_seconds * 2 ** (failures - 1),
                )
            state.seat_failures[seat_id] = failures
            state.failed_until[seat_id] = current_time + wait
            if exitcode == EXIT_CONNECTION_LOST:
                say(
                    seat_id,
                    f"Lost the connection to that game for a while; retrying that game in {wait:.0f}s "
                    "if it is still assigned.",
                )
            elif exitcode == EXIT_NOT_CONNECTED:  # and the connect window has closed
                say(
                    seat_id,
                    f"Couldn't reach that game (a temporary problem); retrying in {wait:.0f}s "
                    "if it is still assigned.",
                )
            else:
                # The game itself goes on (its timers may play for the agent).
                say(
                    seat_id,
                    f"Your agent's process for this match stopped with an error (exit code {exitcode}); "
                    f"the match continues. Retrying in {wait:.0f}s if it is still assigned.",
                )
    active_matches = {match_of(seat_id) for seat_id in state.registry.keys()}
    for match in finished:
        # The match is over: a player of it that had stopped earlier needs no
        # word of its own.
        for seat_id in [s for s in state.stopped if match_of(s) == match]:
            del state.stopped[seat_id]
        if match not in active_matches:  # said once, when its last player here is done
            log("Match finished.")
    if reaped and idle and len(state.registry) == 0:
        say_waiting()

    if state.discovery_retry_at is not None and current_time < state.discovery_retry_at:
        return
    try:
        assignments = official.assignments()
    except PlatformError as exc:
        if not state.discovery_failing:
            log(f"Could not check tournament assignments ({exc}); will keep retrying.")
        state.discovery_failing = True
        return
    except AuthenticationError as exc:
        if is_registration_incomplete(exc):
            # Games already running keep playing until they need a fresh
            # agent session (within about an hour: the backend rechecks the
            # registration at every sign-in); new ones wait until the
            # contestant has done what the message says.
            _note_registration(state, current_time, registration_wait_message(exc), log)
            state.discovery_retry_at = current_time + REGISTRATION_RETRY_SECONDS
            state.discovery_failing = True
            return
        if is_fatal_auth_error(exc):
            raise
        delay = _auth_retry_delay(exc, state.auth_failures)
        state.auth_failures += 1
        state.discovery_retry_at = current_time + delay
        if not state.discovery_failing:
            log(
                f"Could not renew your agent session ({exc}). Games already running keep playing; "
                "retrying automatically."
            )
        state.discovery_failing = True
        return
    state.auth_failures = 0
    state.discovery_retry_at = None
    if state.discovery_failing:
        log("Assignment discovery recovered.")
        state.discovery_failing = False

    listed = {assignment.seat_id for assignment in assignments}
    # A seat that has left the list is done with: forget its failure count
    # and its connect window.
    for seat_id in [
        s for s in state.seat_failures if s not in listed and not state.registry.is_active(s)
    ]:
        del state.seat_failures[seat_id]
    for seat_id in [
        s for s in state.connect_by if s not in listed and not state.registry.is_active(s)
    ]:
        del state.connect_by[seat_id]
    # A game whose worker had stopped (and wasn't started again) has ended:
    # say so once per match, or nothing would mark the end of it.
    ended = [s for s in state.stopped if s not in listed and not state.registry.is_active(s)]
    for seat_id in ended:
        assignment = state.stopped.pop(seat_id)
        game = _clean(assignment.game_type)
        say(
            seat_id,
            f"Your {game + ' ' if game else ''}match{_kind_label(assignment)} has ended while your "
            "agent's process for it was stopped.",
        )
    if ended and len(state.registry) == 0:
        say_waiting()
    for seat_id in [
        s for s in state.seat_games if s not in listed and not state.registry.is_active(s)
    ]:
        del state.seat_games[seat_id]

    # The seats to start now, grouped by match: one set of lines per match,
    # however many of its players this runtime plays (self-play).
    to_start: dict[str, list[OfficialAssignment]] = {}
    for assignment in [] if state.key_refused else assignments:
        seat_id = assignment.seat_id
        if not seat_id or state.registry.is_active(seat_id):
            continue
        only = _only_games()
        if only and (assignment.game_type or "").strip().lower() not in only:
            _note_other_game(assignment, state, log)
            continue
        if assignment_kind(assignment) not in kinds:
            _note_other_kind(assignment, state, log)
            continue
        window = (assignment.match_id, assignment.connect_deadline_at)
        if state.connect_by.get(seat_id, (None, 0.0))[0] != window:
            state.connect_by[seat_id] = (window, _connect_by(assignment, current_time, wall_now()))
        retry_at = state.failed_until.get(seat_id)
        if retry_at is not None and current_time < retry_at:
            continue
        to_start.setdefault(assignment.match_id or seat_id, []).append(assignment)
    session_token = getattr(official, "session_token", None)
    for group in to_start.values():
        for assignment in group:
            seat_id = assignment.seat_id
            worker_input = TournamentWorkerInput(
                seat_id=seat_id,
                match_id=assignment.match_id,
                game_type=assignment.game_type,
                agent_spec=agent_spec,
                execution_id=state.execution_id,
                tournament_id=assignment.tournament_id,
                agent_session=session_token() if callable(session_token) else None,
            )
            process = process_factory(
                target=_tournament_process_entry, args=(worker_input,), daemon=True
            )
            process.start()
            state.registry.start(seat_id, process)
            state.started_at[seat_id] = current_time
            state.seat_games[seat_id] = assignment
            state.stopped.pop(seat_id, None)
        for line in describe_assignment(group[0], now=wall_now(), players=len(group)):
            log(line)
        log("Starting match...")

    for seat_id in state.registry.keys():
        if seat_id in listed:
            state.missing_polls.pop(seat_id, None)
            continue
        state.missing_polls[seat_id] = state.missing_polls.get(seat_id, 0) + 1
        if state.missing_polls[seat_id] >= MISSING_POLLS_BEFORE_STOP:
            say(seat_id, "Match is no longer assigned; stopped its worker.")
            state.registry.terminate(seat_id, shutdown_join_timeout)
            state.missing_polls.pop(seat_id, None)
            state.started_at.pop(seat_id, None)
            state.seat_games.pop(seat_id, None)
            if len(state.registry) == 0:
                say_waiting()


def run_tournament_forever(
    official: OfficialAgentClient,
    *,
    agent_spec: str,
    kinds: Collection[str] = ALL_KINDS,
    poll_interval: float = DEFAULT_TOURNAMENT_POLL_SECONDS,
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
    shutdown_join_timeout: float = DEFAULT_SHUTDOWN_JOIN_TIMEOUT_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
    process_factory: Callable[..., multiprocessing.process.BaseProcess] = _MP_CONTEXT.Process,
    log: Callable[[str], None] = print_line,
    max_iterations: int | None = None,
) -> None:
    """Keep one worker per active official assignment of a kind in ``kinds``
    (``TESTING``, ``TOURNAMENT`` or both; see ``run_tournament_once``) until
    interrupted. Between polls it only sleeps: waiting costs no AI tokens.
    Never exits just because there are no assignments, nor on a temporary
    control-plane or sign-in problem, nor on an incomplete registration (it
    says what to do and keeps trying). A game's worker finding the key refused
    (rotated while running) stops new games only; the runtime itself stops
    when its own sign-in is refused. However it stops (Ctrl+C, the platform
    refusing the key, or ``max_iterations`` in tests), every running worker
    is terminated in ``finally`` — no match is resigned.
    """
    state = TournamentState()
    ticks = 0
    try:
        while max_iterations is None or ticks < max_iterations:
            run_tournament_once(
                official,
                state,
                agent_spec=agent_spec,
                kinds=kinds,
                now=now,
                cooldown_seconds=cooldown_seconds,
                shutdown_join_timeout=shutdown_join_timeout,
                process_factory=process_factory,
                log=log,
            )
            ticks += 1
            sleep(poll_interval)
    finally:
        if len(state.registry) > 0:
            log(f"Stopping {len(state.registry)} active match worker(s)...")
            state.registry.terminate_all(shutdown_join_timeout)
