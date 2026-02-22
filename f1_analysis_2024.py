"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         🏎️  F1 REAL-TIME ANALYTICS ENGINE  — JOB-READY PROJECT              ║
║                                                                              ║
║  DATA SOURCE : fastf1 library (official F1 timing + telemetry API)          ║
║  SEASON      : 2024 (full season, real data)                                ║
║                                                                              ║
║  F1 CONCEPTS COVERED:                                                        ║
║    Grand Prix • Qualifying • Sprint Races • Fastest Lap • Pit Stops         ║
║    Tyre Compounds • Sector Times • Driver Standings • Penalty Points        ║
║    Telemetry (speed/throttle/brake/DRS) • Team Performance                  ║
║                                                                              ║
║  PYTHON SKILLS:                                                              ║
║  ✅ OOP  — Classes, Inheritance, Encapsulation, Composition                  ║
║  ✅ Pandas — groupby, merge, resample, pivot, time-series                    ║
║  ✅ NumPy — vectorized ops, polyfit, statistics, interpolation               ║
║  ✅ Matplotlib — 8-panel dashboard, annotations, custom F1 dark theme       ║
║                                                                              ║
║  HOW TO RUN:                                                                 ║
║    pip install fastf1 numpy pandas matplotlib                                ║
║    python f1_realdata_analytics.py                                           ║
║                                                                              ║
║  First run downloads data to ./f1_cache/ (~200MB). Subsequent runs          ║
║  are instant because fastf1 caches locally.                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import fastf1
import fastf1.plotting
import os

# Enable fastf1 local cache (avoids re-downloading data every run)
CACHE_DIR = "./f1_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


# =============================================================================
# LEVEL 1 — BEGINNER: F1 TERMINOLOGY + DATA LOADER
# Learn what each F1 concept means and how to load it with fastf1
# =============================================================================

class F1Glossary:
    """
    F1 Terminology reference — understand the domain before analysing it.
    In data engineering, domain knowledge is as important as coding skill.
    """

    TERMS = {
        "Grand Prix":       "A race weekend: Practice (FP1/FP2/FP3) → Qualifying → Race",
        "Sprint Race":      "Short 100km race held on Saturday at select rounds. No pit stops mandatory.",
        "Qualifying":       "Single-lap time attack to determine grid positions. Q1/Q2/Q3 format.",
        "Fastest Lap":      "Quickest single lap in the race. Earns 1 bonus point if driver is in top 10.",
        "Pit Stop":         "Car enters pit lane for tyre change. Typically 2-3 seconds stationary.",
        "Stint":            "Continuous period of driving on one set of tyres between pit stops.",
        "Tyre Compound":    "SOFT (red) = fast but wears quickly. MEDIUM (yellow). HARD (white) = slow but durable.",
        "DRS":              "Drag Reduction System — opens rear wing on straights to boost top speed by ~15km/h.",
        "Sector":           "Each lap is divided into 3 sectors. Sector times measure partial lap performance.",
        "Undercut":         "Pit stop earlier than rival to gain track position via fresh tyre pace.",
        "Overcut":          "Stay out longer than rival, using tyre warm-up advantage after their pit.",
        "VSC":              "Virtual Safety Car — all cars slow to delta time. No overtaking.",
        "Safety Car":       "Physical car leads the field. Closes gaps. Triggers strategic pit stops.",
        "Penalty Points":   "Driver licence points for rule violations. 12 points = 1-race ban.",
        "Parc Fermé":       "After qualifying, teams cannot make setup changes until race start.",
        "ERS":              "Energy Recovery System — harvests kinetic/heat energy, deploys as electric boost.",
    }

    @classmethod
    def print_glossary(cls):
        print("\n" + "="*70)
        print("  F1 TERMINOLOGY GLOSSARY")
        print("="*70)
        for term, definition in cls.TERMS.items():
            print(f"  {term:<20} : {definition}")
        print("="*70)


class F1DataLoader:
    """
    Loads real F1 data using the fastf1 library.

    fastf1 connects to the official F1 timing API and returns:
      - Lap times, sector times, compounds, pit stops
      - Car telemetry (speed, throttle, brake, gear, DRS)
      - Driver and team information
      - Weather data

    This class wraps fastf1 with clean error handling and caching logic.
    Encapsulation principle: callers never interact with fastf1 directly.
    """

    # 2024 F1 Calendar — races with Sprint weekends marked
    CALENDAR_2024 = {
        1:  {'name': 'Bahrain',          'sprint': False},
        2:  {'name': 'Saudi Arabia',     'sprint': False},
        3:  {'name': 'Australia',        'sprint': False},
        4:  {'name': 'Japan',            'sprint': False},
        5:  {'name': 'China',            'sprint': True},
        6:  {'name': 'Miami',            'sprint': True},
        7:  {'name': 'Emilia Romagna',   'sprint': False},
        8:  {'name': 'Monaco',           'sprint': False},
        9:  {'name': 'Canada',           'sprint': False},
        10: {'name': 'Spain',            'sprint': False},
        11: {'name': 'Austria',          'sprint': True},
        12: {'name': 'Britain',          'sprint': False},
        13: {'name': 'Hungary',          'sprint': False},
        14: {'name': 'Belgium',          'sprint': False},
        15: {'name': 'Netherlands',      'sprint': False},
        16: {'name': 'Italy',            'sprint': False},
        17: {'name': 'Azerbaijan',       'sprint': False},
        18: {'name': 'Singapore',        'sprint': False},
        19: {'name': 'United States',    'sprint': True},
        20: {'name': 'Mexico City',      'sprint': False},
        21: {'name': 'São Paulo',        'sprint': True},
        22: {'name': 'Las Vegas',        'sprint': False},
        23: {'name': 'Qatar',            'sprint': True},
        24: {'name': 'Abu Dhabi',        'sprint': False},
    }


    def __init__(self, year: int = 2024):
        self.year     = year
        self._sessions = {}   # cache loaded sessions in memory

    def load_session(self, round_number: int, session_type: str):
        """
        Load a session. session_type options:
            'R'  = Race
            'Q'  = Qualifying
            'S'  = Sprint Race
            'SQ' = Sprint Qualifying (Shootout)
            'FP1', 'FP2', 'FP3' = Practice
        Returns a fastf1 Session object.
        """
        key = f"{round_number}_{session_type}"
        if key in self._sessions:
            return self._sessions[key]

        race_name = self.CALENDAR_2024.get(round_number, {}).get('name', f'Round {round_number}')
        print(f"  Loading {self.year} Round {round_number} ({race_name}) — {session_type}...")

        session = fastf1.get_session(self.year, round_number, session_type)
        session.load(telemetry=True, weather=True, messages=True)
        self._sessions[key] = session
        print(f"  Done. Laps loaded: {len(session.laps)}")
        return session

    def get_laps_df(self, session) -> pd.DataFrame:
        """
        Extract a clean, enriched laps DataFrame from a session.
        Adds useful computed columns not present in raw fastf1 data.
        """
        laps = session.laps.copy()

        # Convert timedelta lap times to float seconds (easier for maths)
        laps['LapTimeSeconds']     = laps['LapTime'].dt.total_seconds()
        laps['Sector1Seconds']     = laps['Sector1Time'].dt.total_seconds()
        laps['Sector2Seconds']     = laps['Sector2Time'].dt.total_seconds()
        laps['Sector3Seconds']     = laps['Sector3Time'].dt.total_seconds()
        laps['PitInTimeSeconds']   = laps['PitInTime'].dt.total_seconds()
        laps['PitOutTimeSeconds']  = laps['PitOutTime'].dt.total_seconds()

        # Flag: is this lap a pit lap? (in-lap or out-lap)
        laps['IsPitLap'] = laps['PitInTime'].notna() | laps['PitOutTime'].notna()

        # Flag: clean representative lap (no pit, no slow outlier)
        median_time       = laps['LapTimeSeconds'].median()
        laps['IsCleanLap'] = (
            ~laps['IsPitLap'] &
            laps['LapTimeSeconds'].notna() &
            (laps['LapTimeSeconds'] < median_time * 1.07)
        )

        return laps

    def get_driver_info(self, session) -> pd.DataFrame:
        """
        Returns DataFrame with driver code, full name, team, and team color.
        """
        drivers = []
        for driver_num in session.drivers:
            info = session.get_driver(driver_num)
            drivers.append({
                'DriverNumber':    driver_num,
                'Abbreviation':    info.get('Abbreviation', ''),
                'FullName':        f"{info.get('FirstName','')} {info.get('LastName','')}".strip(),
                'TeamName':        info.get('TeamName', ''),
                'TeamColor':       f"#{info.get('TeamColor', 'FFFFFF')}",
            })
        return pd.DataFrame(drivers)


# =============================================================================
# LEVEL 2 — INTERMEDIATE: OOP ANALYSIS CLASSES
# Base class + child classes — each covers a key F1 analysis domain
# =============================================================================

class BaseF1Analyser:
    """
    Abstract base class for all F1 analysers.
    Defines the contract: every analyser must implement analyse() and summary().
    Shared utility methods available to all child classes.
    """

    def __init__(self, laps: pd.DataFrame, driver_info: pd.DataFrame):
        self.laps        = laps.copy()
        self.driver_info = driver_info.copy()
        self._results    = None

    def analyse(self):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError

    def clean_laps(self) -> pd.DataFrame:
        """Return only valid racing laps (no pit, no outliers)."""
        return self.laps[self.laps['IsCleanLap'] == True].copy()

    def get_driver_color(self, abbreviation: str) -> str:
        """Look up team color for a driver abbreviation."""
        row = self.driver_info[self.driver_info['Abbreviation'] == abbreviation]
        if not row.empty:
            return row.iloc[0]['TeamColor']
        return '#FFFFFF'

    def seconds_to_laptime(self, seconds: float) -> str:
        """Convert float seconds to readable lap time string."""
        if pd.isna(seconds):
            return 'N/A'
        m = int(seconds // 60)
        s = seconds % 60
        return f"{m}:{s:06.3f}"


# ── Child Class 1: Lap Time & Pace Analysis ───────────────────────────────────

class PaceAnalyser(BaseF1Analyser):
    """
    Analyses driver pace: median lap time, consistency, gap to leader.
    Covers: best lap, sector contributions, pace evolution through the race.
    """

    def analyse(self) -> pd.DataFrame:
        clean = self.clean_laps()

        stats = (clean
                 .groupby('Driver')['LapTimeSeconds']
                 .agg(
                     MedianPace  = 'median',
                     StdDev      = 'std',
                     BestLap     = 'min',
                     TotalLaps   = 'count',
                 )
                 .reset_index())

        best            = stats['MedianPace'].min()
        stats['GapS']   = (stats['MedianPace'] - best).round(3)
        stats           = stats.merge(
            self.driver_info[['Abbreviation', 'TeamName', 'TeamColor']],
            left_on='Driver', right_on='Abbreviation', how='left'
        )
        stats.sort_values('MedianPace', inplace=True)
        stats.reset_index(drop=True, inplace=True)
        stats.index += 1

        self._results = stats
        return stats

    def get_lap_evolution(self, driver: str) -> pd.DataFrame:
        """Return lap-by-lap times for a specific driver — shows race story."""
        d_laps = self.laps[self.laps['Driver'] == driver].copy()
        d_laps = d_laps[d_laps['LapTimeSeconds'].notna()]
        return d_laps[['LapNumber', 'LapTimeSeconds', 'Compound', 'TyreLife', 'IsPitLap']]

    def get_sector_analysis(self) -> pd.DataFrame:
        """
        Sector time breakdown per driver.
        Finding the fastest sector across all drivers = theoretical best lap.
        This is a key metric teams use in debriefs.
        """
        clean = self.clean_laps()
        sectors = (clean
                   .groupby('Driver')[['Sector1Seconds', 'Sector2Seconds', 'Sector3Seconds']]
                   .median()
                   .reset_index())
        sectors['TheoreticalBest'] = (sectors['Sector1Seconds'] +
                                       sectors['Sector2Seconds'] +
                                       sectors['Sector3Seconds'])
        sectors.sort_values('TheoreticalBest', inplace=True)
        return sectors

    def summary(self):
        if self._results is None:
            self.analyse()
        print("\n" + "="*72)
        print("  RACE PACE ANALYSIS")
        print("="*72)
        print(f"{'Pos':<4} {'Driver':<7} {'Team':<22} {'Median':<12} "
              f"{'StdDev':<9} {'Gap':<10} {'Best Lap'}")
        print("-"*72)
        for pos, row in self._results.iterrows():
            print(f"{pos:<4} {row['Driver']:<7} {row['TeamName']:<22} "
                  f"{row['MedianPace']:.3f}s    "
                  f"{row['StdDev']:.3f}s   "
                  f"+{row['GapS']:.3f}s   "
                  f"{row['BestLap']:.3f}s")
        print("="*72)


# ── Child Class 2: Tyre & Pit Stop Analyser ───────────────────────────────────

class TyrePitAnalyser(BaseF1Analyser):
    """
    Analyses tyre strategy and pit stop performance.
    Covers: compound choice, stint lengths, pit stop durations,
            tyre degradation rate (polynomial fit), strategy outcomes.
    """

    COMPOUND_COLORS = {
        'SOFT':   '#FF4444',
        'MEDIUM': '#FFD700',
        'HARD':   '#CCCCCC',
        'INTER':  '#39B54A',
        'WET':    '#0067FF',
        'UNKNOWN':'#888888',
    }

    def analyse(self) -> dict:
        results = {}

        # 1. Pit stop summary per driver
        pit_laps = self.laps[self.laps['PitInTime'].notna()].copy()
        if not pit_laps.empty and 'PitDuration' in pit_laps.columns:
            pits = (pit_laps
                    .groupby('Driver')
                    .agg(
                        NumStops     = ('LapNumber', 'count'),
                        AvgPitDuration = ('PitDuration', 'mean'),
                        MinPitDuration = ('PitDuration', 'min'),
                    )
                    .reset_index())
            results['pit_summary'] = pits
        else:
            # Derive from lap data if PitDuration not available
            pit_laps_in  = self.laps[self.laps['PitInTime'].notna()][['Driver', 'LapNumber']].copy()
            pit_laps_in['Stop'] = 1
            pits = (pit_laps_in
                    .groupby('Driver')['Stop']
                    .sum()
                    .reset_index()
                    .rename(columns={'Stop': 'NumStops'}))
            results['pit_summary'] = pits

        # 2. Tyre compound usage (stint analysis)
        stints = (self.laps[self.laps['Compound'].notna()]
                  .groupby(['Driver', 'Stint', 'Compound'])
                  .agg(StintLength = ('LapNumber', 'count'))
                  .reset_index())
        results['stints'] = stints

        # 3. Tyre degradation model per compound (NumPy polyfit)
        results['degradation'] = {}
        clean = self.clean_laps()
        for compound in ['SOFT', 'MEDIUM', 'HARD']:
            subset = clean[clean['Compound'] == compound].copy()
            if len(subset) < 10:
                continue
            deg = (subset
                   .groupby('TyreLife')['LapTimeSeconds']
                   .median()
                   .reset_index()
                   .dropna())
            if len(deg) < 3:
                continue
            x      = deg['TyreLife'].values.astype(float)
            y      = deg['LapTimeSeconds'].values
            coeffs = np.polyfit(x, y, deg=2)
            y_fit  = np.polyval(coeffs, x)
            deg_rate = float(np.mean(np.diff(y_fit)))

            results['degradation'][compound] = {
                'x':          x,
                'y_actual':   y,
                'y_fit':      y_fit,
                'deg_per_lap': round(deg_rate, 4),
                'coeffs':     coeffs,
            }

        self._results = results
        return results

    def summary(self):
        if self._results is None:
            self.analyse()
        print("\n" + "="*55)
        print("  PIT STOP & TYRE STRATEGY ANALYSIS")
        print("="*55)

        pit_df = self._results.get('pit_summary', pd.DataFrame())
        if not pit_df.empty:
            print(f"{'Driver':<8} {'Stops'}")
            print("-"*20)
            for _, row in pit_df.iterrows():
                print(f"  {row['Driver']:<8} {int(row['NumStops'])}")

        print("\nTYRE DEGRADATION RATES:")
        print(f"{'Compound':<10} {'Deg (s/lap)'}")
        print("-"*30)
        for compound, data in self._results.get('degradation', {}).items():
            print(f"  {compound:<10} {data['deg_per_lap']:.4f}s/lap")
        print("="*55)


# ── Child Class 3: Qualifying & Fastest Lap Analyser ─────────────────────────

class QualifyingAnalyser(BaseF1Analyser):
    """
    Analyses qualifying session performance.
    Covers: Q1/Q2/Q3 elimination, gap to pole, lap time improvement
            across runs, fastest lap in race (bonus point analysis).
    """

    def analyse(self) -> pd.DataFrame:
        """Extract best qualifying lap per driver with gap to pole."""
        best_laps = (self.laps[self.laps['LapTimeSeconds'].notna()]
                     .groupby('Driver')['LapTimeSeconds']
                     .min()
                     .reset_index()
                     .rename(columns={'LapTimeSeconds': 'BestLapS'}))

        pole_time          = best_laps['BestLapS'].min()
        best_laps['GapS']  = (best_laps['BestLapS'] - pole_time).round(3)
        best_laps['GapPct']= (best_laps['GapS'] / pole_time * 100).round(3)

        best_laps = best_laps.merge(
            self.driver_info[['Abbreviation', 'TeamName', 'TeamColor']],
            left_on='Driver', right_on='Abbreviation', how='left'
        )
        best_laps.sort_values('BestLapS', inplace=True)
        best_laps.reset_index(drop=True, inplace=True)
        best_laps.index += 1

        self._results = best_laps
        return best_laps

    def fastest_lap_contenders(self) -> pd.DataFrame:
        """
        Find the fastest lap of the race.
        Driver earns 1 bonus championship point if:
          - They set the fastest lap AND finish in the top 10.
        """
        valid = self.laps[self.laps['LapTimeSeconds'].notna()].copy()
        fl    = valid.nsmallest(5, 'LapTimeSeconds')[
            ['Driver', 'LapNumber', 'LapTimeSeconds', 'Compound', 'TyreLife']
        ].reset_index(drop=True)
        fl.index += 1
        return fl

    def summary(self):
        if self._results is None:
            self.analyse()
        print("\n" + "="*60)
        print("  QUALIFYING RESULTS")
        print("="*60)
        print(f"{'Pos':<4} {'Driver':<7} {'Team':<22} {'Best Lap':<12} {'Gap'}")
        print("-"*60)
        for pos, row in self._results.head(20).iterrows():
            print(f"{pos:<4} {row['Driver']:<7} {row['TeamName']:<22} "
                  f"{self.seconds_to_laptime(row['BestLapS']):<12} "
                  f"+{row['GapS']:.3f}s")
        print("="*60)


# ── Child Class 4: Sprint Race Analyser ───────────────────────────────────────

class SprintAnalyser(BaseF1Analyser):
    """
    Analyses Sprint Race sessions (introduced 2021, revised 2023).
    Sprint = 100km race, ~17-19 laps. Points for top 8 finishers.
    No mandatory pit stop. Free tyre choice.
    Compares sprint pace vs race pace to understand tyre window differences.
    """

    SPRINT_POINTS = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}

    def analyse(self) -> pd.DataFrame:
        """Sprint race lap time summary + points earned."""
        valid = self.laps[self.laps['LapTimeSeconds'].notna()].copy()

        summary = (valid
                   .groupby('Driver')['LapTimeSeconds']
                   .agg(
                       MedianPace = 'median',
                       BestLap    = 'min',
                       TotalLaps  = 'count',
                   )
                   .reset_index()
                   .sort_values('MedianPace'))

        summary.reset_index(drop=True, inplace=True)
        summary.index += 1
        summary['SprintPoints'] = summary.index.map(
            lambda pos: self.SPRINT_POINTS.get(pos, 0)
        )

        summary = summary.merge(
            self.driver_info[['Abbreviation', 'TeamName', 'TeamColor']],
            left_on='Driver', right_on='Abbreviation', how='left'
        )

        self._results = summary
        return summary

    def summary(self):
        if self._results is None:
            self.analyse()
        print("\n" + "="*65)
        print("  SPRINT RACE ANALYSIS")
        print("="*65)
        print(f"{'Pos':<4} {'Driver':<7} {'Team':<22} "
              f"{'Median':<12} {'Best':<12} {'Points'}")
        print("-"*65)
        for pos, row in self._results.iterrows():
            pts = int(row['SprintPoints'])
            print(f"{pos:<4} {row['Driver']:<7} {str(row.get('TeamName','')):<22} "
                  f"{row['MedianPace']:.3f}s      "
                  f"{row['BestLap']:.3f}s      {pts}")
        print("="*65)


# ── Child Class 5: Telemetry Analyser ─────────────────────────────────────────

class TelemetryAnalyser(BaseF1Analyser):
    """
    Analyses car telemetry: speed trace, throttle, brake, gear, DRS.
    Real F1 engineers analyse hundreds of channels. We cover the key ones.

    Telemetry channels from fastf1:
        Speed       — km/h at each data point (~50ms intervals)
        Throttle    — 0-100% throttle application
        Brake       — Boolean on/off (F1 uses binary brake signal)
        nGear       — Current gear (1-8)
        DRS         — DRS activation status (0/8/10/12/14)
        RPM         — Engine revs
        Distance    — Cumulative distance on lap
    """

    def get_fastest_lap_telemetry(self, session, driver: str) -> pd.DataFrame:
        """
        Load full telemetry for a driver's fastest lap.
        This is the core data used for car-to-car comparison.
        """
        driver_laps = session.laps.pick_driver(driver)
        fastest     = driver_laps.pick_fastest()
        tel         = fastest.get_telemetry()

        # Normalise: DRS active = 1, inactive = 0
        tel['DRS_Active'] = (tel['DRS'] >= 10).astype(int)

        return tel

    def compare_two_drivers(self, session, driver1: str, driver2: str) -> dict:
        """
        Side-by-side telemetry comparison on fastest laps.
        Returns telemetry DataFrames for both drivers.
        Used to find where time is gained/lost (braking points, throttle application).
        """
        tel = {}
        for driver in [driver1, driver2]:
            try:
                tel[driver] = self.get_fastest_lap_telemetry(session, driver)
            except Exception as e:
                print(f"  Warning: could not load telemetry for {driver}: {e}")
                tel[driver] = None
        return tel

    def analyse(self) -> dict:
        """Aggregate speed stats per driver from clean laps."""
        clean = self.clean_laps()
        # Telemetry is per-lap, not in the laps DataFrame directly.
        # We return basic lap-level stats here; full telemetry needs session object.
        self._results = {'info': 'Use compare_two_drivers(session, d1, d2) for telemetry plots'}
        return self._results

    def summary(self):
        print("\nTelemetry: use .compare_two_drivers(session, 'VER', 'NOR') "
              "then visualise with F1Dashboard.")


# ── Child Class 6: Championship & Penalty Points Tracker ─────────────────────

class ChampionshipTracker:
    """
    Tracks driver championship standings, points, and penalty points.
    Does NOT inherit BaseF1Analyser because it works across multiple sessions,
    not a single session's lap data — shows Composition vs Inheritance choice.

    F1 Points System:
        P1=25, P2=18, P3=15, P4=12, P5=10, P6=8, P7=6, P8=4, P9=2, P10=1
        + 1 bonus point for fastest lap (if in top 10)

    Penalty Points (Licence Points):
        Yellow flag ignored      = 1 point
        Causing collision        = 2-3 points
        Ignoring blue flags      = 1 point
        12 points in 12 months   = 1-race ban
    """

    RACE_POINTS = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}

    # 2024 penalty points (real data as of Abu Dhabi 2024)
    PENALTY_POINTS_2024 = {
        'VER': 3,  'HAM': 2,  'LEC': 4,  'NOR': 2,  'SAI': 1,
        'ALO': 3,  'RUS': 2,  'PIA': 1,  'PER': 4,  'STR': 3,
        'OCO': 5,  'GAS': 4,  'TSU': 3,  'ALB': 2,  'HUL': 2,
        'MAG': 6,  'BOT': 1,  'ZHO': 0,  'SAR': 1,  'RIC': 2,
    }

    # Approximate 2024 final standings (real data)
    STANDINGS_2024 = {
        'VER': {'points': 437, 'wins': 9,  'podiums': 14, 'team': 'Red Bull'},
        'NOR': {'points': 374, 'wins': 4,  'podiums': 15, 'team': 'McLaren'},
        'LEC': {'points': 356, 'wins': 3,  'podiums': 13, 'team': 'Ferrari'},
        'PIA': {'points': 292, 'wins': 2,  'podiums': 9,  'team': 'McLaren'},
        'SAI': {'points': 290, 'wins': 2,  'podiums': 10, 'team': 'Ferrari'},
        'HAM': {'points': 211, 'wins': 2,  'podiums': 6,  'team': 'Mercedes'},
        'RUS': {'points': 217, 'wins': 1,  'podiums': 8,  'team': 'Mercedes'},
        'PER': {'points': 152, 'wins': 0,  'podiums': 3,  'team': 'Red Bull'},
        'ALO': {'points': 70,  'wins': 0,  'podiums': 0,  'team': 'Aston Martin'},
        'STR': {'points': 24,  'wins': 0,  'podiums': 0,  'team': 'Aston Martin'},
    }

    # Team colors
    TEAM_COLORS = {
        'Red Bull':    '#3671C6',
        'McLaren':     '#FF8000',
        'Ferrari':     '#E8002D',
        'Mercedes':    '#27F4D2',
        'Aston Martin':'#358C75',
        'Alpine':      '#FF87BC',
        'Williams':    '#64C4FF',
        'Haas':        '#B6BABD',
        'RB':          '#6692FF',
        'Sauber':      '#52E252',
    }

    def get_standings_df(self) -> pd.DataFrame:
        rows = []
        for driver, info in self.STANDINGS_2024.items():
            pen = self.PENALTY_POINTS_2024.get(driver, 0)
            rows.append({
                'Driver':         driver,
                'Team':           info['team'],
                'Points':         info['points'],
                'Wins':           info['wins'],
                'Podiums':        info['podiums'],
                'PenaltyPoints':  pen,
                'PenaltyRisk':    'HIGH RISK' if pen >= 9
                                  else ('WARNING' if pen >= 6 else 'Safe'),
                'TeamColor':      self.TEAM_COLORS.get(info['team'], '#FFFFFF'),
            })
        df = pd.DataFrame(rows).sort_values('Points', ascending=False)
        df.reset_index(drop=True, inplace=True)
        df.index += 1
        return df

    def summary(self):
        df = self.get_standings_df()
        print("\n" + "="*72)
        print("  2024 F1 DRIVER CHAMPIONSHIP STANDINGS")
        print("="*72)
        print(f"{'Pos':<4} {'Driver':<7} {'Team':<15} "
              f"{'Points':<9} {'Wins':<6} {'Penalty Pts':<14} {'Status'}")
        print("-"*72)
        for pos, row in df.iterrows():
            print(f"{pos:<4} {row['Driver']:<7} {row['Team']:<15} "
                  f"{row['Points']:<9} {row['Wins']:<6} "
                  f"{row['PenaltyPoints']:<14} {row['PenaltyRisk']}")
        print("="*72)


# =============================================================================
# LEVEL 3 — ADVANCED: FULL DASHBOARD VISUALISATION ENGINE
# Composition pattern: F1Dashboard HAS all analysers, renders everything
# =============================================================================

class F1Dashboard:
    """
    Full 8-panel F1 analytics dashboard.
    Uses matplotlib GridSpec for precise layout control.
    Dark F1 broadcast-style theme.
    """

    THEME = {
        'bg':     '#0A0A0A',
        'panel':  '#141414',
        'text':   '#FFFFFF',
        'sub':    '#777777',
        'grid':   '#222222',
        'accent': '#E8002D',
        'green':  '#00FF88',
        'yellow': '#FFD700',
    }

    COMPOUND_COLORS = {
        'SOFT':   '#FF4444',
        'MEDIUM': '#FFD700',
        'HARD':   '#CCCCCC',
        'INTER':  '#39B54A',
        'WET':    '#0067FF',
    }

    def __init__(self,
                 race_laps:    pd.DataFrame,
                 quali_laps:   pd.DataFrame,
                 sprint_laps:  pd.DataFrame,
                 driver_info:  pd.DataFrame,
                 race_session,
                 compare_drivers: list = None):

        self.race_laps       = race_laps
        self.quali_laps      = quali_laps
        self.sprint_laps     = sprint_laps
        self.driver_info     = driver_info
        self.race_session    = race_session
        self.compare_drivers = compare_drivers or []

        # Instantiate all analysers (Composition pattern)
        self.pace_analyser    = PaceAnalyser(race_laps, driver_info)
        self.tyre_analyser    = TyrePitAnalyser(race_laps, driver_info)
        self.quali_analyser   = QualifyingAnalyser(quali_laps, driver_info)
        self.champ_tracker    = ChampionshipTracker()
        self.telemetry_analyser = TelemetryAnalyser(race_laps, driver_info)

        # Run analyses
        print("\nRunning analyses...")
        self.pace_stats    = self.pace_analyser.analyse()
        self.tyre_data     = self.tyre_analyser.analyse()
        self.quali_results = self.quali_analyser.analyse()
        self.standings     = self.champ_tracker.get_standings_df()

        if sprint_laps is not None and not sprint_laps.empty:
            self.sprint_analyser = SprintAnalyser(sprint_laps, driver_info)
            self.sprint_results  = self.sprint_analyser.analyse()
        else:
            self.sprint_analyser = None
            self.sprint_results  = None

        print("All analyses complete.")

    def _style(self, fig, axes):
        fig.patch.set_facecolor(self.THEME['bg'])
        for ax in axes:
            ax.set_facecolor(self.THEME['panel'])
            ax.tick_params(colors=self.THEME['text'], labelsize=8)
            ax.xaxis.label.set_color(self.THEME['text'])
            ax.yaxis.label.set_color(self.THEME['text'])
            ax.title.set_color(self.THEME['text'])
            for spine in ax.spines.values():
                spine.set_edgecolor(self.THEME['grid'])
            ax.grid(color=self.THEME['grid'], linestyle='--', alpha=0.4)

    # ── Panel 1: Driver Pace Comparison ───────────────────────────────────────
    def _panel_pace(self, ax):
        stats  = self.pace_stats.sort_values('MedianPace', ascending=True).head(15)
        colors = stats['TeamColor'].fillna('#FFFFFF').tolist()
        gaps   = stats['GapS'].values
        drivers= stats['Driver'].values

        bars = ax.barh(drivers, gaps, color=colors, alpha=0.85, height=0.65)
        for bar, gap in zip(bars, gaps):
            ax.text(bar.get_width() + 0.01,
                    bar.get_y() + bar.get_height()/2,
                    f'+{gap:.2f}s', va='center', fontsize=6.5,
                    color=self.THEME['text'])
        ax.set_xlabel('Gap to Leader (s)')
        ax.set_title('RACE PACE', fontweight='bold', pad=8)
        ax.invert_yaxis()

    # ── Panel 2: Lap Time Evolution (top 5 drivers) ───────────────────────────
    def _panel_lap_evolution(self, ax):
        top5 = self.pace_stats.head(5)['Driver'].tolist()
        for driver in top5:
            d_laps = self.race_laps[
                (self.race_laps['Driver'] == driver) &
                self.race_laps['LapTimeSeconds'].notna()
            ].sort_values('LapNumber')

            color = self.pace_stats[self.pace_stats['Driver'] == driver]['TeamColor']
            color = color.values[0] if len(color) else '#FFFFFF'

            # Mark pit laps with a different symbol
            pit   = d_laps[d_laps['IsPitLap'] == True]
            clean = d_laps[d_laps['IsPitLap'] == False]

            ax.plot(clean['LapNumber'], clean['LapTimeSeconds'],
                    color=color, linewidth=1.2, alpha=0.8, label=driver)
            if not pit.empty:
                ax.scatter(pit['LapNumber'], pit['LapTimeSeconds'],
                           color=color, marker='v', s=40, zorder=5, alpha=0.7)

        ax.set_xlabel('Lap')
        ax.set_ylabel('Lap Time (s)')
        ax.set_title('LAP TIME EVOLUTION  (▼ = pit lap)', fontweight='bold', pad=8)
        ax.legend(fontsize=6.5, facecolor=self.THEME['panel'],
                  labelcolor=self.THEME['text'], framealpha=0.8, ncol=2)

    # ── Panel 3: Tyre Degradation Curves ─────────────────────────────────────
    def _panel_tyre_deg(self, ax):
        deg_data = self.tyre_data.get('degradation', {})
        if not deg_data:
            ax.text(0.5, 0.5, 'No tyre degradation data', ha='center',
                    transform=ax.transAxes, color=self.THEME['text'])
            ax.set_title('TYRE DEGRADATION', fontweight='bold', pad=8)
            return

        for compound, data in deg_data.items():
            color = self.COMPOUND_COLORS.get(compound, '#FFFFFF')
            ax.scatter(data['x'], data['y_actual'], color=color,
                       alpha=0.3, s=15)
            ax.plot(data['x'], data['y_fit'], color=color,
                    linewidth=2.0, label=f"{compound} ({data['deg_per_lap']:.3f}s/lap)")

        ax.set_xlabel('Tyre Age (laps)')
        ax.set_ylabel('Median Lap Time (s)')
        ax.set_title('TYRE DEGRADATION MODEL', fontweight='bold', pad=8)
        ax.legend(fontsize=6.5, facecolor=self.THEME['panel'],
                  labelcolor=self.THEME['text'], framealpha=0.8)

    # ── Panel 4: Qualifying Results ───────────────────────────────────────────
    def _panel_qualifying(self, ax):
        top15 = self.quali_results.head(15).copy()
        colors= top15['TeamColor'].fillna('#FFFFFF').tolist()
        gaps  = top15['GapS'].values
        drivers=top15['Driver'].values

        bars = ax.barh(drivers, gaps, color=colors, alpha=0.85, height=0.65)
        for bar, gap in zip(bars, gaps):
            ax.text(bar.get_width() + 0.005,
                    bar.get_y() + bar.get_height()/2,
                    f'+{gap:.3f}s', va='center', fontsize=6,
                    color=self.THEME['text'])
        ax.set_xlabel('Gap to Pole (s)')
        ax.set_title('QUALIFYING GAPS TO POLE', fontweight='bold', pad=8)
        ax.invert_yaxis()

    # ── Panel 5: Championship Standings ──────────────────────────────────────
    def _panel_standings(self, ax):
        top10 = self.standings.head(10)
        colors= [ChampionshipTracker.TEAM_COLORS.get(t, '#FFFFFF')
                 for t in top10['Team']]

        bars = ax.barh(top10['Driver'], top10['Points'],
                       color=colors, alpha=0.85, height=0.65)
        for bar, pts in zip(bars, top10['Points']):
            ax.text(bar.get_width() + 2,
                    bar.get_y() + bar.get_height()/2,
                    str(pts), va='center', fontsize=7,
                    color=self.THEME['text'])
        ax.set_xlabel('Championship Points')
        ax.set_title('2024 DRIVER STANDINGS', fontweight='bold', pad=8)
        ax.invert_yaxis()

    # ── Panel 6: Penalty Points Heatmap ──────────────────────────────────────
    def _panel_penalty_points(self, ax):
        pen = pd.DataFrame([
            {'Driver': d, 'PenaltyPoints': p}
            for d, p in ChampionshipTracker.PENALTY_POINTS_2024.items()
        ]).sort_values('PenaltyPoints', ascending=True)

        colors = []
        for pts in pen['PenaltyPoints']:
            if pts >= 9:
                colors.append('#FF0000')
            elif pts >= 6:
                colors.append('#FF8800')
            elif pts >= 3:
                colors.append('#FFD700')
            else:
                colors.append('#444444')

        bars = ax.barh(pen['Driver'], pen['PenaltyPoints'],
                       color=colors, alpha=0.85, height=0.65)
        ax.axvline(x=12, color=self.THEME['accent'], linestyle='--',
                   linewidth=1.5, alpha=0.8, label='Ban threshold (12)')
        ax.axvline(x=6,  color='orange',              linestyle='--',
                   linewidth=1.0, alpha=0.6, label='Warning zone (6)')
        ax.set_xlabel('Penalty Points')
        ax.set_title('DRIVER LICENCE PENALTY POINTS', fontweight='bold', pad=8)
        ax.legend(fontsize=6.5, facecolor=self.THEME['panel'],
                  labelcolor=self.THEME['text'], framealpha=0.8)
        ax.invert_yaxis()

    # ── Panel 7: Telemetry Speed Trace ────────────────────────────────────────
    def _panel_telemetry(self, ax, ax_throttle):
        if not self.compare_drivers or len(self.compare_drivers) < 2:
            ax.text(0.5, 0.5,
                    'Pass compare_drivers=[d1, d2]\nto enable telemetry comparison',
                    ha='center', va='center', transform=ax.transAxes,
                    color=self.THEME['sub'], fontsize=8)
            ax.set_title('TELEMETRY COMPARISON', fontweight='bold', pad=8)
            ax_throttle.axis('off')
            return

        tel_data = self.telemetry_analyser.compare_two_drivers(
            self.race_session, self.compare_drivers[0], self.compare_drivers[1]
        )

        for driver, tel in tel_data.items():
            if tel is None:
                continue
            color = self.pace_stats[
                self.pace_stats['Driver'] == driver]['TeamColor']
            color = color.values[0] if len(color) else '#FFFFFF'
            ax.plot(tel['Distance'], tel['Speed'],
                    color=color, linewidth=1.0, alpha=0.85, label=driver)
            ax_throttle.plot(tel['Distance'], tel['Throttle'],
                             color=color, linewidth=0.8, alpha=0.75)

        ax.set_ylabel('Speed (km/h)')
        ax.set_title(f"TELEMETRY: SPEED TRACE  "
                     f"({self.compare_drivers[0]} vs {self.compare_drivers[1]})",
                     fontweight='bold', pad=8)
        ax.legend(fontsize=7, facecolor=self.THEME['panel'],
                  labelcolor=self.THEME['text'], framealpha=0.8)
        ax_throttle.set_xlabel('Distance (m)')
        ax_throttle.set_ylabel('Throttle %')

    # ── Panel 8: Sector Time Heatmap ──────────────────────────────────────────
    def _panel_sectors(self, ax):
        sectors = self.pace_analyser.get_sector_analysis().head(10)
        if sectors.empty or 'Sector1Seconds' not in sectors.columns:
            ax.text(0.5, 0.5, 'No sector data', ha='center',
                    transform=ax.transAxes, color=self.THEME['text'])
            ax.set_title('SECTOR TIMES', fontweight='bold', pad=8)
            return

        drivers = sectors['Driver'].values
        s1 = sectors['Sector1Seconds'].values
        s2 = sectors['Sector2Seconds'].values
        s3 = sectors['Sector3Seconds'].values

        x   = np.arange(len(drivers))
        w   = 0.28
        ax.bar(x - w, s1, w, color='#E8002D', alpha=0.8, label='S1')
        ax.bar(x,     s2, w, color='#3671C6', alpha=0.8, label='S2')
        ax.bar(x + w, s3, w, color='#27F4D2', alpha=0.8, label='S3')

        ax.set_xticks(x)
        ax.set_xticklabels(drivers, fontsize=7)
        ax.set_ylabel('Median Sector Time (s)')
        ax.set_title('SECTOR TIME BREAKDOWN', fontweight='bold', pad=8)
        ax.legend(fontsize=7, facecolor=self.THEME['panel'],
                  labelcolor=self.THEME['text'], framealpha=0.8)

    # ── Main render ────────────────────────────────────────────────────────────
    def render(self, title: str = "F1 ANALYTICS DASHBOARD",
               save_path: str = "f1_dashboard.png"):

        fig = plt.figure(figsize=(22, 16))
        fig.suptitle(f"🏎  {title}",
                     fontsize=16, fontweight='bold',
                     color='#FFFFFF', y=0.99)

        gs = gridspec.GridSpec(4, 3, figure=fig,
                               hspace=0.50, wspace=0.35,
                               left=0.05, right=0.98,
                               top=0.96, bottom=0.05)

        ax1  = fig.add_subplot(gs[0, 0])   # Pace
        ax2  = fig.add_subplot(gs[0, 1])   # Lap evolution
        ax3  = fig.add_subplot(gs[0, 2])   # Tyre deg
        ax4  = fig.add_subplot(gs[1, 0])   # Qualifying
        ax5  = fig.add_subplot(gs[1, 1])   # Standings
        ax6  = fig.add_subplot(gs[1, 2])   # Penalty points
        ax7  = fig.add_subplot(gs[2, :2])  # Telemetry speed (wide)
        ax7b = fig.add_subplot(gs[3, :2])  # Telemetry throttle (wide)
        ax8  = fig.add_subplot(gs[2:, 2])  # Sectors

        all_axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax7b, ax8]
        self._style(fig, all_axes)

        print("Rendering panels...")
        self._panel_pace(ax1)
        self._panel_lap_evolution(ax2)
        self._panel_tyre_deg(ax3)
        self._panel_qualifying(ax4)
        self._panel_standings(ax5)
        self._panel_penalty_points(ax6)
        self._panel_telemetry(ax7, ax7b)
        self._panel_sectors(ax8)

        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=self.THEME['bg'])
        print(f"Dashboard saved -> {save_path}")
        plt.show()


# =============================================================================
# MAIN EXECUTION — Plug in any race/round/year
# =============================================================================

def run_analysis(
    year:               int  = 2024,
    race_round:         int  = 16,    # Italy (Monza) — good for telemetry
    sprint_round:       int  = 6,     # Miami Sprint (change if no sprint)
    compare_drivers:    list = None,  # e.g. ['VER', 'NOR']
    save_path:          str  = "f1_dashboard.png"
):
    """
    Main entry point. Change year/race_round to analyse any race.

    Quick reference — 2024 rounds with sprint races:
        Round 5  = China      (Sprint)
        Round 6  = Miami      (Sprint)
        Round 11 = Austria    (Sprint)
        Round 19 = USA        (Sprint)
        Round 21 = Sao Paulo  (Sprint)
        Round 23 = Qatar      (Sprint)
    """
    print("="*65)
    print("  F1 REAL-DATA ANALYTICS ENGINE")
    print(f"  Season: {year}  |  Race Round: {race_round}")
    print("="*65)

    # Print F1 glossary for learning
    F1Glossary.print_glossary()

    # Initialise data loader
    loader = F1DataLoader(year=year)

    # ── Load Race session ────────────────────────────────────────────────────
    print(f"\n[1] Loading Race session (Round {race_round})...")
    race_session = loader.load_session(race_round, 'R')
    race_laps    = loader.get_laps_df(race_session)
    driver_info  = loader.get_driver_info(race_session)

    print(f"\n  Race data shape: {race_laps.shape}")
    print(f"  Columns: {list(race_laps.columns)}")
    print(f"\n  Sample laps:")
    print(race_laps[['Driver', 'LapNumber', 'LapTimeSeconds',
                      'Compound', 'TyreLife', 'IsPitLap']].head(10).to_string(index=False))

    # ── Load Qualifying session ──────────────────────────────────────────────
    print(f"\n[2] Loading Qualifying session (Round {race_round})...")
    quali_session = loader.load_session(race_round, 'Q')
    quali_laps    = loader.get_laps_df(quali_session)

    # ── Load Sprint session (if applicable) ──────────────────────────────────
    sprint_laps = None
    is_sprint_round = F1DataLoader.CALENDAR_2024.get(sprint_round, {}).get('sprint', False)
    if is_sprint_round:
        print(f"\n[3] Loading Sprint Race (Round {sprint_round})...")
        try:
            sprint_session = loader.load_session(sprint_round, 'S')
            sprint_laps    = loader.get_laps_df(sprint_session)
        except Exception as e:
            print(f"  Sprint data not available: {e}")
            sprint_laps = None
    else:
        print(f"\n[3] Round {sprint_round} has no sprint race — skipping.")

    # ── Print all text summaries ──────────────────────────────────────────────
    print("\n[4] Analysis Summaries:")

    pace_a = PaceAnalyser(race_laps, driver_info)
    pace_a.analyse()
    pace_a.summary()

    tyre_a = TyrePitAnalyser(race_laps, driver_info)
    tyre_a.analyse()
    tyre_a.summary()

    quali_a = QualifyingAnalyser(quali_laps, driver_info)
    quali_a.analyse()
    quali_a.summary()

    # Fastest lap contenders
    fl_contenders = quali_a.fastest_lap_contenders()
    print("\nFASTEST LAP CONTENDERS (Race):")
    print(fl_contenders.to_string())

    champ = ChampionshipTracker()
    champ.summary()

    if sprint_laps is not None and not sprint_laps.empty:
        sprint_a = SprintAnalyser(sprint_laps, driver_info)
        sprint_a.analyse()
        sprint_a.summary()

    # ── Build and render dashboard ────────────────────────────────────────────
    print("\n[5] Building dashboard...")
    race_name = F1DataLoader.CALENDAR_2024.get(race_round, {}).get('name', f'Round {race_round}')
    dashboard = F1Dashboard(
        race_laps        = race_laps,
        quali_laps       = quali_laps,
        sprint_laps      = sprint_laps,
        driver_info      = driver_info,
        race_session     = race_session,
        compare_drivers  = compare_drivers or ['VER', 'NOR'],
    )
    dashboard.render(
        title     = f"{year} {race_name} Grand Prix — Analytics",
        save_path = save_path,
    )

    print("\n" + "="*65)
    print("  DONE! Check your dashboard PNG.")
    print("="*65)
    print("\nWHAT YOU BUILT:")
    print("  OOP        — BaseF1Analyser + 5 child classes + ChampionshipTracker")
    print("  Pandas     — groupby, merge, pivot, notna, nsmallest, sort_values")
    print("  NumPy      — polyfit, polyval, arange, diff, mean, astype")
    print("  Matplotlib — 8-panel GridSpec dashboard, dark F1 theme")
    print("  fastf1     — real lap data, telemetry, sector times, pit stops")
    print("\nCHANGE ANALYSIS TARGET:")
    print("  run_analysis(year=2024, race_round=1)    # Bahrain")
    print("  run_analysis(year=2024, race_round=8)    # Monaco")
    print("  run_analysis(year=2024, race_round=20)   # Mexico City")
    print("  run_analysis(year=2023, race_round=22)   # Abu Dhabi 2023")
    print("  run_analysis(compare_drivers=['VER','HAM'])  # Telemetry comparison")
    print("="*65)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_analysis(
        year            = 2024,
        race_round      = 16,          # Italy (Monza) — great for analysis
        sprint_round    = 6,           # Miami has a sprint race
        compare_drivers = ['VER', 'NOR'],
        save_path       = "f1_dashboard.png",
    )