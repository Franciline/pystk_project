"""Extra world-state diagnostics; never part of the policy observation vector."""
import numpy as np


class EpisodeDiagnostics:
    def __init__(self, env, settings=None):
        self.env = env
        self.settings = dict(settings or {})
        self.step_size = float(env.step_size)
        self.seconds = float(self.settings.get('stuck_seconds', 5))
        self.radius = float(self.settings.get('stuck_radius_m', 1))
        if min(self.step_size, self.seconds, self.radius) <= 0:
            raise ValueError('Diagnostic time and distance thresholds must be positive')
        self.rows = []
        self.anchor = None
        self.stuck = False
        self.stuck_count = self.rescue_count = 0
        self.finished_stuck_seconds = 0.
        self.previous_rescue = False
        self.capture()

    def _end_stuck(self):
        if self.stuck:
            self.finished_stuck_seconds += self.rows[-1]['elapsed_seconds'] - self.anchor['elapsed_seconds']
        self.anchor, self.stuck = None, False

    def capture(self):
        world = self.env.world
        kart = world.karts[self.env.kart_ix]
        rescue = getattr(kart, 'rescue_active', None)
        rescue = bool(rescue) if rescue is not None else None
        row = dict(elapsed_seconds=len(self.rows)*self.step_size,
                   race_time_s=float(world.time), position=np.asarray(kart.location).tolist(),
                   progress_m=float(kart.overall_distance), rescue_active=rescue,
                   racing=float(world.time) > 0, stuck_active=False)
        if rescue is True and not self.previous_rescue:
            self.rescue_count += 1
        self.previous_rescue = rescue is True
        if not row['racing'] or rescue:
            if self.rows:
                self._end_stuck()
        else:
            if self.anchor is not None:
                moved = np.linalg.norm(np.asarray(row['position'])-self.anchor['position']) > self.radius
                progressed = abs(row['progress_m']-self.anchor['progress_m']) > self.radius
                if moved or progressed:
                    self._end_stuck()
            if self.anchor is None:
                self.anchor = row
            if row['elapsed_seconds']-self.anchor['elapsed_seconds'] >= self.seconds-1e-8:
                if not self.stuck:
                    self.stuck_count += 1
                self.stuck = row['stuck_active'] = True
        self.rows.append(row)
        return row

    def summary(self):
        duration = self.finished_stuck_seconds
        if self.stuck:
            duration += self.rows[-1]['elapsed_seconds']-self.anchor['elapsed_seconds']
        supported = all(row['rescue_active'] is not None for row in self.rows)
        return dict(rescue_status_available=supported,
                    rescue_count=self.rescue_count if supported else None,
                    stuck_count=self.stuck_count, stuck_duration_s=round(duration, 6))
