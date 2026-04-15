#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import copy
import json
import math
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
TRACKING_DIR = ROOT / "tracking"
DATA_DIR = ROOT / "data"
CONFIG_PATH = ROOT / "edge_model_config.json"
BET_RESULTS_PATH = TRACKING_DIR / "bet_results.csv"
RECOMMENDATIONS_LOG_PATH = TRACKING_DIR / "daily_recommendations.csv"
MODEL_RESULTS_PATH = TRACKING_DIR / "model_results.csv"
EXTERNAL_MARKET_FEED_PATH = DATA_DIR / "external_market_feed.json"
PROVIDER_MARKET_FEED_PATH = DATA_DIR / "provider_market_feed.json"
PROVIDER_CREDENTIALS_PATH = DATA_DIR / "provider_credentials.json"
CLOSING_LINE_OVERRIDES_PATH = TRACKING_DIR / "closing_line_overrides.csv"
SETTLEMENT_OVERRIDES_PATH = TRACKING_DIR / "settlement_overrides.csv"
PROVIDER_SYNC_REPORT_PATH = OUTPUT_DIR / "provider-sync-latest.json"
PHONE_CARD_ROOT_PATH = ROOT / "PHONE - Last Desktop Card.html"
PHONE_CARD_TEXT_ROOT_PATH = ROOT / "PHONE - Last Desktop Card.txt"
HISTORICAL_DATA_PATH = DATA_DIR / "historical_data.json"
ML_MODELS_PATH = DATA_DIR / "ml_models.pkl"

SPORT_ENDPOINTS = {
    "MLB": "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={date}",
    "NCAABASE": "https://site.api.espn.com/apis/site/v2/sports/baseball/college-baseball/scoreboard?dates={date}",
    "NBA": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date}",
    "WNBA": "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard?dates={date}",
    "NCAAMB": "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?dates={date}",
    "NCAAWB": "https://site.api.espn.com/apis/site/v2/sports/basketball/womens-college-basketball/scoreboard?dates={date}",
    "NFL": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates={date}",
    "NCAAF": "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?dates={date}",
    "NHL": "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard?dates={date}",
    "MLS": "https://site.api.espn.com/apis/site/v2/sports/soccer/usa.1/scoreboard?dates={date}",
    "EPL": "https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard?dates={date}",
    "La Liga": "https://site.api.espn.com/apis/site/v2/sports/soccer/esp.1/scoreboard?dates={date}",
    "Bundesliga": "https://site.api.espn.com/apis/site/v2/sports/soccer/ger.1/scoreboard?dates={date}",
    "Serie A": "https://site.api.espn.com/apis/site/v2/sports/soccer/ita.1/scoreboard?dates={date}",
    "Ligue 1": "https://site.api.espn.com/apis/site/v2/sports/soccer/fra.1/scoreboard?dates={date}",
    "UCL": "https://site.api.espn.com/apis/site/v2/sports/soccer/uefa.champions/scoreboard?dates={date}",
}

MARKET_HAIRCUTS = {
    "moneyline": 0.35,
    "spread": 0.45,
    "total": 0.50,
    "player_prop": 0.80,
    "team_prop": 0.65,
    "other": 0.70,
}

BASEBALL_SPORTS = {"MLB", "NCAABASE"}
BASKETBALL_SPORTS = {"NBA", "WNBA", "NCAAMB", "NCAAWB"}
FOOTBALL_SPORTS = {"NFL", "NCAAF"}
HOCKEY_SPORTS = {"NHL"}
SOCCER_SPORTS = {"MLS", "EPL", "La Liga", "Bundesliga", "Serie A", "Ligue 1", "UCL"}
SPREAD_MODEL_SPORTS = BASKETBALL_SPORTS | FOOTBALL_SPORTS | HOCKEY_SPORTS
PROVIDER_REPORT_EMPTY = {
    "provider": "",
    "enabled": False,
    "synced": False,
    "requested_date": "",
    "generated_at": "",
    "sports_seen": 0,
    "events_seen": 0,
    "markets_written": 0,
    "target_books_used": [],
    "warnings": [],
    "feed_path": str(PROVIDER_MARKET_FEED_PATH),
    "requests_remaining": "",
    "requests_used": "",
    "last_request_cost": "",
}
ODDS_API_SPORT_DISPLAY_MAP = {
    "americanfootball_nfl": "NFL",
    "americanfootball_ncaaf": "NCAAF",
    "baseball_mlb": "MLB",
    "baseball_ncaa": "NCAABASE",
    "basketball_nba": "NBA",
    "basketball_wnba": "WNBA",
    "basketball_ncaab": "NCAAMB",
    "basketball_wncaab": "NCAAWB",
    "icehockey_nhl": "NHL",
    "soccer_usa_mls": "MLS",
    "soccer_epl": "EPL",
    "soccer_esp_la_liga": "La Liga",
    "soccer_ger_bundesliga": "Bundesliga",
    "soccer_ita_serie_a": "Serie A",
    "soccer_fra_ligue_1": "Ligue 1",
    "soccer_uefa_champions_league": "UCL",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a daily betting card from public ESPN boards.")
    parser.add_argument("--date", default=date.today().strftime("%Y%m%d"), help="Date in YYYYMMDD format")
    parser.add_argument("--top", type=int, default=0, help="Maximum number of bets to keep; 0 means no hard cap")
    parser.add_argument("--book", default="", help="Sportsbook label for the output")
    parser.add_argument("--skip-provider-sync", action="store_true", help="Skip the live odds provider sync even if configured")
    parser.add_argument("--force-provider-sync", action="store_true", help="Ignore provider cache and request a fresh live sync")
    args = parser.parse_args()

    config = load_json(CONFIG_PATH)
    book_label = args.book.strip() or str(config.get("default_sportsbook", "BetOrigin")).strip() or "BetOrigin"
    ensure_tracking_files()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    day_iso = normalize_iso_date(args.date)
    provider_report = sync_provider_market_feed(day_iso, config, force_refresh=args.force_provider_sync) if not args.skip_provider_sync else skipped_provider_report(day_iso, "Provider sync skipped by CLI flag.")
    effective_config = build_runtime_config(config, provider_report)
    settle_model_results(day_iso)
    ledger_state = load_ledger_state(effective_config)
    history_profile = load_history_profile(effective_config)

    candidates: list[dict[str, Any]] = []
    scoreboard_payloads: dict[str, dict[str, Any]] = {}
    for sport, url_template in SPORT_ENDPOINTS.items():
        payload = fetch_json(url_template.format(date=args.date))
        scoreboard_payloads[sport] = payload
        candidates.extend(build_sport_candidates(sport, payload, effective_config, ledger_state, history_profile, book_label, day_iso))
    candidates.extend(load_external_feed_candidates(day_iso, effective_config, ledger_state, history_profile, scoreboard_payloads, book_label))
    candidates = deduplicate_candidate_sources(candidates)

    candidates.sort(key=lambda item: item["composite_score"], reverse=True)
    shortlist = pick_top_candidates(candidates, effective_config, args.top)
    
    # Store historical data for ML training
    store_historical_data(candidates, day_iso)
    
    # Train ML models if enough historical data exists
    historical_data = load_historical_data()
    if historical_data:
        print(f"Attempting to train ML models with {len(historical_data)} days of data...")
        training_results = train_ml_models_from_historical_data()
        if training_results.get("success"):
            print("ML models trained successfully")
        else:
            print(f"ML training skipped: {training_results.get('error', 'Unknown error')}")
    else:
        print("ML training skipped: No historical data available yet")

    latest_candidates_path = OUTPUT_DIR / "daily-candidates-latest.json"
    dated_candidates_path = OUTPUT_DIR / f"daily-candidates-{day_iso}.json"
    latest_card_json_path = OUTPUT_DIR / "daily-card-latest.json"
    dated_card_json_path = OUTPUT_DIR / f"daily-card-{day_iso}.json"
    latest_card_md_path = OUTPUT_DIR / "daily-card-latest.md"
    dated_card_md_path = OUTPUT_DIR / f"daily-card-{day_iso}.md"
    latest_card_html_path = OUTPUT_DIR / "daily-card-latest.html"
    dated_card_html_path = OUTPUT_DIR / f"daily-card-{day_iso}.html"
    latest_phone_html_path = OUTPUT_DIR / "phone-card-latest.html"
    dated_phone_html_path = OUTPUT_DIR / f"phone-card-{day_iso}.html"
    latest_phone_text_path = OUTPUT_DIR / "phone-card-latest.txt"
    dated_phone_text_path = OUTPUT_DIR / f"phone-card-{day_iso}.txt"
    latest_performance_json_path = OUTPUT_DIR / "model-performance-latest.json"
    dated_performance_json_path = OUTPUT_DIR / f"model-performance-{day_iso}.json"
    latest_provider_report_path = OUTPUT_DIR / "provider-sync-latest.json"
    dated_provider_report_path = OUTPUT_DIR / f"provider-sync-{day_iso}.json"

    write_json(latest_candidates_path, candidates)
    write_json(dated_candidates_path, candidates)
    write_json(latest_card_json_path, [automation_shape(item) for item in shortlist])
    write_json(dated_card_json_path, [automation_shape(item) for item in shortlist])

    append_recommendations_log(shortlist, day_iso)
    sync_model_results_snapshot(shortlist, day_iso)
    settle_model_results(day_iso)
    model_performance = load_model_performance()

    write_json(latest_performance_json_path, model_performance)
    write_json(dated_performance_json_path, model_performance)
    write_json(latest_provider_report_path, provider_report)
    write_json(dated_provider_report_path, provider_report)

    readable = build_readable_card(shortlist, candidates, ledger_state, model_performance, provider_report, effective_config, day_iso)
    latest_card_md_path.write_text(readable, encoding="utf-8")
    dated_card_md_path.write_text(readable, encoding="utf-8")
    html_report = build_html_report(shortlist, candidates, ledger_state, model_performance, provider_report, effective_config, day_iso)
    latest_card_html_path.write_text(html_report, encoding="utf-8")
    dated_card_html_path.write_text(html_report, encoding="utf-8")
    phone_html_report = build_phone_html_report(shortlist, candidates, ledger_state, model_performance, provider_report, effective_config, day_iso)
    latest_phone_html_path.write_text(phone_html_report, encoding="utf-8")
    dated_phone_html_path.write_text(phone_html_report, encoding="utf-8")
    PHONE_CARD_ROOT_PATH.write_text(phone_html_report, encoding="utf-8")
    # Also update index.html for GitHub Pages / phone access
    INDEX_HTML_PATH = ROOT / "index.html"
    INDEX_HTML_PATH.write_text(phone_html_report, encoding="utf-8")
    phone_text_report = build_phone_text_card(shortlist, candidates, ledger_state, model_performance, provider_report, effective_config, day_iso)
    latest_phone_text_path.write_text(phone_text_report, encoding="utf-8")
    dated_phone_text_path.write_text(phone_text_report, encoding="utf-8")
    PHONE_CARD_TEXT_ROOT_PATH.write_text(phone_text_report, encoding="utf-8")

    print(f"Generated {len(candidates)} candidates and {len(shortlist)} final picks for {day_iso}.")
    print(f"Model tracking: {model_performance['settled_bets']} settled, {model_performance['open_bets']} open, ROI {model_performance['roi_pct']:.2f}%.")
    print(f"Provider feed: {'synced' if provider_report['synced'] else 'not synced'} via {provider_report.get('provider') or 'none'} with {provider_report['markets_written']} rows.")
    print(f"Candidates: {dated_candidates_path}")
    print(f"Card JSON: {dated_card_json_path}")
    print(f"Card Markdown: {dated_card_md_path}")
    print(f"Card HTML: {dated_card_html_path}")
    print(f"Phone HTML: {dated_phone_html_path}")
    print(f"Phone text: {dated_phone_text_path}")
    print(f"Model performance JSON: {dated_performance_json_path}")
    return 0


def build_sport_candidates(
    sport: str,
    payload: dict[str, Any],
    config: dict[str, Any],
    ledger_state: dict[str, float],
    history_profile: dict[str, Any],
    book: str,
    requested_date_iso: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for event in payload.get("events", []):
        competition = safe_get(event, "competitions", 0)
        if not competition:
            continue
        odds = safe_get(competition, "odds", 0)
        if not odds:
            continue

        home = extract_team_context(competition, "home", sport)
        away = extract_team_context(competition, "away", sport)
        if not home or not away:
            continue

        model = build_model(sport, event, home, away, config)
        if not model:
            continue

        event_id = str(event.get("id", ""))
        event_name = event.get("name", "")
        sample_size = int(min(home["games"], away["games"]))
        drawdown_multiplier = config["drawdown_stake_multiplier"] if ledger_state["current_drawdown_pct"] >= config["drawdown_trigger_pct"] else 1.0
        current_bankroll = max(0.0, ledger_state["current_bankroll"])

        candidates.extend(
            build_market_candidates(
                sport=sport,
                event_name=event_name,
                event_id=event_id,
                event_date=requested_date_iso,
                odds_block=odds,
                model=model,
                sample_size=sample_size,
                config=config,
                drawdown_multiplier=drawdown_multiplier,
                current_bankroll=current_bankroll,
                history_profile=history_profile,
                book=book,
                home=home,
                away=away,
            )
        )
    return candidates


def load_external_feed_candidates(
    day_iso: str,
    config: dict[str, Any],
    ledger_state: dict[str, float],
    history_profile: dict[str, Any],
    scoreboard_payloads: dict[str, dict[str, Any]],
    default_book: str,
) -> list[dict[str, Any]]:
    drawdown_multiplier = config["drawdown_stake_multiplier"] if ledger_state["current_drawdown_pct"] >= config["drawdown_trigger_pct"] else 1.0
    current_bankroll = max(0.0, ledger_state["current_bankroll"])
    scoreboard_index = build_scoreboard_model_index(scoreboard_payloads, config)
    candidates: list[dict[str, Any]] = []

    for source_name, path in (("external_feed", EXTERNAL_MARKET_FEED_PATH), ("provider_feed", PROVIDER_MARKET_FEED_PATH)):
        if not path.exists():
            continue
        raw_payload = load_json(path)
        feed_rows = raw_payload.get("markets", []) if isinstance(raw_payload, dict) else raw_payload
        if not isinstance(feed_rows, list):
            continue

        for row in feed_rows:
            if not isinstance(row, dict):
                continue
            row_date = normalize_iso_date(str(row.get("date", "")))
            if row_date != day_iso:
                continue

            sport = str(row.get("sport", "")).strip().upper()
            event_name = str(row.get("event", "")).strip()
            bet_type = str(row.get("bet_type", "")).strip()
            if not sport or not event_name or not bet_type:
                continue

            odds_format = str(row.get("odds_format", "")).strip().lower()
            odds = parse_odds_to_american(row.get("odds"), odds_format)
            if odds is None:
                continue
            opposite_odds = parse_odds_to_american(row.get("opposite_odds"), odds_format)
            fair_probability = parse_probability(row.get("fair_probability"))
            if fair_probability is None and opposite_odds is not None:
                fair_probability = devig_probability(odds, opposite_odds)
            if fair_probability is None:
                fair_probability = american_to_probability(odds)

            market_type = str(row.get("market_type", "other")).strip().lower() or "other"
            if market_type not in MARKET_HAIRCUTS:
                market_type = "other"
            quality = parse_numeric(row.get("quality"))
            sample_size = parse_numeric(row.get("sample_size"))
            event_id = str(row.get("event_id", "")).strip() or slugify(f"{day_iso}-{sport}-{event_name}-{bet_type}")
            line = str(row.get("line", "")).strip()
            notes = str(row.get("notes", "")).strip() or "Imported external market feed."
            sportsbook = str(row.get("sportsbook", "")).strip() or default_book
            home_team = str(row.get("home_team", "")).strip()
            away_team = str(row.get("away_team", "")).strip()
            team_key = matchup_key(home_team, away_team) if home_team and away_team else ""
            model_context = (
                scoreboard_index.get((sport, event_id))
                or scoreboard_index.get((sport, slugify(event_name)))
                or (scoreboard_index.get((sport, team_key)) if team_key else None)
            )
            consensus_probability = parse_probability(row.get("consensus_probability"))
            sharp_probability = parse_probability(row.get("sharp_probability"))
            model_probability = parse_probability(row.get("model_probability"))
            derived_probability = derive_external_model_probability(row, model_context)
            if model_probability is None:
                model_probability = blend_external_model_probability(
                    consensus_probability=consensus_probability,
                    sharp_probability=sharp_probability,
                    derived_probability=derived_probability,
                    model_context=model_context,
                    config=config,
                )
            if model_probability is None:
                continue
            if math.isnan(quality):
                quality = external_feed_quality(row, model_context)
            if math.isnan(sample_size):
                sample_size = external_feed_sample_size(row, model_context)

            correlation_group = f"{sport}:{team_key}" if team_key else f"{sport}:{slugify(event_name)}"
            feature_home = model_context["home"] if model_context else None
            feature_away = model_context["away"] if model_context else None
            feature_model = model_context["model"] if model_context else None
            feature_line_value = parse_numeric(line)
            candidate = evaluate_market_candidate(
                event_date=day_iso,
                sport=sport,
                event_name=event_name,
                event_id=event_id,
                market_type=market_type,
                bet_type=bet_type,
                odds=odds,
                opposite_odds=opposite_odds,
                fair_probability=fair_probability,
                model_probability=model_probability,
                quality=quality if not math.isnan(quality) else 72.0,
                sample_size=int(sample_size) if not math.isnan(sample_size) else 25,
                config=config,
                drawdown_multiplier=drawdown_multiplier,
                current_bankroll=current_bankroll,
                history_profile=history_profile,
                book=sportsbook,
                line=line,
                notes=notes,
                correlation_group=correlation_group,
                consensus_probability=consensus_probability,
                sharp_probability=sharp_probability,
                reference_book_count=safe_int(row.get("reference_book_count")),
                sharp_book_count=safe_int(row.get("sharp_book_count")),
                home_team=home_team or (feature_home["team"] if feature_home else ""),
                away_team=away_team or (feature_away["team"] if feature_away else ""),
                selection=str(row.get("selection", "")).strip().lower(),
                selection_team=str(row.get("selection_team", "")).strip(),
                training_features=build_candidate_feature_vector(
                    sport=sport,
                    market_type=market_type,
                    sample_size=int(sample_size) if not math.isnan(sample_size) else 25,
                    quality=quality if not math.isnan(quality) else 72.0,
                    fair_probability=fair_probability,
                    model_probability=model_probability,
                    odds=odds,
                    opposite_odds=opposite_odds,
                    line_value=None if math.isnan(feature_line_value) else feature_line_value,
                    home=feature_home,
                    away=feature_away,
                    model=feature_model,
                ),
            )
            candidate["source"] = source_name
            candidate["providerConsensusProbability"] = "" if consensus_probability is None else round(consensus_probability * 100.0, 2)
            candidate["providerSharpProbability"] = "" if sharp_probability is None else round(sharp_probability * 100.0, 2)
            candidates.append(candidate)

    return candidates


def build_scoreboard_model_index(
    scoreboard_payloads: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for sport, payload in scoreboard_payloads.items():
        for event in payload.get("events", []):
            competition = safe_get(event, "competitions", 0)
            if not competition:
                continue
            home = extract_team_context(competition, "home", sport)
            away = extract_team_context(competition, "away", sport)
            if not home or not away:
                continue
            model = build_model(sport, event, home, away, config)
            if not model:
                continue
            event_id = str(event.get("id", "")).strip()
            event_name = str(event.get("name", "")).strip()
            sample_size = int(min(home["games"], away["games"]))
            record = {
                "event": event,
                "home": home,
                "away": away,
                "model": model,
                "sample_size": sample_size,
            }
            if event_id:
                index[(sport, event_id)] = record
            if event_name:
                index[(sport, slugify(event_name))] = record
            matchup = matchup_key(home["team"], away["team"])
            if matchup:
                index[(sport, matchup)] = record
    return index


def skipped_provider_report(day_iso: str, reason: str, provider_name: str = "", enabled: bool = False) -> dict[str, Any]:
    preserved = load_preserved_provider_report(day_iso, reason, provider_name=provider_name, enabled=enabled)
    if preserved:
        return preserved
    generated_at = datetime.now().isoformat(timespec="seconds")
    write_json(
        PROVIDER_MARKET_FEED_PATH,
        {
            "provider": provider_name,
            "requested_date": day_iso,
            "generated_at": generated_at,
            "markets": [],
        },
    )
    report = dict(PROVIDER_REPORT_EMPTY)
    report.update(
        {
            "provider": provider_name,
            "enabled": enabled,
            "requested_date": day_iso,
            "generated_at": generated_at,
            "warnings": [reason] if reason else [],
        }
    )
    return report


def sync_provider_market_feed(day_iso: str, config: dict[str, Any], force_refresh: bool = False) -> dict[str, Any]:
    provider_cfg = config.get("provider_feed", {})
    provider_name = str(provider_cfg.get("provider", "")).strip().lower()
    if not provider_cfg.get("enabled", False):
        return skipped_provider_report(day_iso, "Provider feed disabled in config.", provider_name)
    if provider_name != "the_odds_api":
        return skipped_provider_report(day_iso, f"Unsupported provider '{provider_name}'.", provider_name, enabled=True)

    api_key = load_provider_api_key(provider_cfg)
    if not api_key:
        return skipped_provider_report(day_iso, f"Missing provider API key. Set {provider_cfg.get('api_key_env', 'THE_ODDS_API_KEY')} or create data/provider_credentials.json.", provider_name, enabled=True)

    cached_report = None if force_refresh else load_cached_provider_report(day_iso, provider_cfg)
    if cached_report:
        return cached_report

    base_url = str(provider_cfg.get("base_url", "https://api.the-odds-api.com")).rstrip("/")
    requested_regions = [str(item).strip() for item in provider_cfg.get("regions_priority", ["us"]) if str(item).strip()]
    if not requested_regions:
        requested_regions = ["us"]

    report = dict(PROVIDER_REPORT_EMPTY)
    report.update(
        {
            "provider": provider_name,
            "enabled": True,
            "requested_date": day_iso,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "feed_path": str(PROVIDER_MARKET_FEED_PATH),
        }
    )

    request_usage = {"requests_remaining": "", "requests_used": "", "last_request_cost": ""}

    try:
        sports_payload, sports_headers = fetch_json_with_headers(
            build_provider_url(
                base_url,
                "/v4/sports",
                {
                    "apiKey": api_key,
                },
            )
        )
        update_provider_usage(request_usage, sports_headers)
    except Exception as exc:
        return skipped_provider_report(day_iso, f"Provider sports request failed: {exc}", provider_name, enabled=True)

    if not isinstance(sports_payload, list):
        return skipped_provider_report(day_iso, "Provider sports response was not a list.", provider_name, enabled=True)

    requested_markets = [item for item in provider_cfg.get("markets", ["h2h", "spreads", "totals"]) if isinstance(item, str) and item.strip()]
    local_timezone = str(provider_cfg.get("timezone", "America/Chicago")).strip() or "America/Chicago"
    max_sports = int(provider_cfg.get("max_sports_per_run", 0) or 0)
    sports = [sport for sport in sports_payload if odds_api_sport_is_supported(sport, requested_markets)]
    priority_sports = [str(item).strip() for item in provider_cfg.get("priority_sports", []) if str(item).strip()]
    sports.sort(key=lambda item: provider_sport_sort_key(item, priority_sports))
    if max_sports > 0:
        sports = sports[:max_sports]

    markets: list[dict[str, Any]] = []
    target_books_used: set[str] = set()
    warnings: list[str] = []
    events_seen = 0
    sports_seen = 0
    prefilter_events = bool(provider_cfg.get("prefilter_events", True))
    rate_limit_failures = 0
    max_rate_limit_failures = int(provider_cfg.get("max_rate_limit_failures", 3))
    request_pause_ms = int(provider_cfg.get("request_pause_ms", 0))
    retry_rate_limit_count = int(provider_cfg.get("retry_rate_limit_count", 1))
    retry_rate_limit_pause_ms = int(provider_cfg.get("retry_rate_limit_pause_ms", 1000))

    for sport_index, sport in enumerate(sports):
        sport_key = str(sport.get("key", "")).strip()
        if not sport_key:
            continue
        todays_event_ids: list[str] = []
        if prefilter_events:
            try:
                event_payload, event_headers = fetch_provider_json_with_retry(
                    build_provider_url(
                        base_url,
                        f"/v4/sports/{sport_key}/events",
                        {
                            "apiKey": api_key,
                            "dateFormat": "iso",
                        },
                    ),
                    pause_ms=request_pause_ms,
                    retry_count=retry_rate_limit_count,
                    retry_pause_ms=retry_rate_limit_pause_ms,
                )
                update_provider_usage(request_usage, event_headers)
            except Exception as exc:
                warnings.append(f"{sport_key}: events request failed ({exc})")
                continue
            if not isinstance(event_payload, list):
                warnings.append(f"{sport_key}: events response was not a list")
                continue
            for event in event_payload:
                if odds_api_local_date(event.get("commence_time"), local_timezone) != day_iso:
                    continue
                event_id = str(event.get("id", "")).strip()
                if event_id:
                    todays_event_ids.append(event_id)
            if not todays_event_ids:
                continue

        sports_seen += 1
        remaining_credits = safe_int(request_usage.get("requests_remaining"), default=-1)
        if remaining_credits == 0:
            warnings.append("Provider quota exhausted before odds sync completed.")
            break
        market_groups = build_provider_market_groups(
            requested_markets=requested_markets,
            region_count=max(1, len(requested_regions)),
            remaining_credits=remaining_credits if remaining_credits >= 0 else None,
            remaining_sports=max(1, len(sports) - sport_index),
        )
        if not market_groups:
            warnings.append("Provider quota too low to request any odds markets.")
            break

        seen_event_ids: set[str] = set()
        for market_group in market_groups:
            current_remaining = safe_int(request_usage.get("requests_remaining"), default=-1)
            estimated_cost = max(1, len(requested_regions) * len(market_group))
            if current_remaining >= 0 and current_remaining < estimated_cost:
                continue
            query = {
                "apiKey": api_key,
                "regions": ",".join(requested_regions),
                "markets": ",".join(market_group),
                "oddsFormat": "american",
                "dateFormat": "iso",
            }
            if todays_event_ids:
                query["eventIds"] = ",".join(todays_event_ids)
            try:
                events, odds_headers = fetch_provider_json_with_retry(
                    build_provider_url(base_url, f"/v4/sports/{sport_key}/odds", query),
                    pause_ms=request_pause_ms,
                    retry_count=retry_rate_limit_count,
                    retry_pause_ms=retry_rate_limit_pause_ms,
                )
                update_provider_usage(request_usage, odds_headers)
            except Exception as exc:
                warnings.append(f"{sport_key}: odds request failed ({exc})")
                if "429" in str(exc):
                    rate_limit_failures += 1
                    if rate_limit_failures >= max_rate_limit_failures:
                        warnings.append("Provider sync stopped early after repeated rate-limit failures.")
                        break
                break
            if not isinstance(events, list):
                warnings.append(f"{sport_key}: odds response was not a list")
                continue
            rate_limit_failures = 0

            for event in events:
                local_event_date = odds_api_local_date(event.get("commence_time"), local_timezone)
                if local_event_date != day_iso:
                    continue
                event_rows = normalize_odds_api_event(event, sport, market_group, provider_cfg, day_iso)
                if not event_rows:
                    continue
                event_id = str(event.get("id", "")).strip()
                if event_id and event_id not in seen_event_ids:
                    seen_event_ids.add(event_id)
                    events_seen += 1
                for row in event_rows:
                    target_key = str(row.get("target_bookmaker_key", "")).strip()
                    if target_key:
                        target_books_used.add(target_key)
                markets.extend(event_rows)
        if rate_limit_failures >= max_rate_limit_failures:
            break

    provider_payload = {
        "provider": provider_name,
        "requested_date": day_iso,
        "generated_at": report["generated_at"],
        "markets": markets,
    }
    if not markets and warnings:
        preserved = load_preserved_provider_report(
            day_iso,
            "Live provider sync failed, so the app is preserving the last same-day provider feed.",
            provider_name=provider_name,
            enabled=True,
        )
        if preserved:
            return preserved
    write_json(PROVIDER_MARKET_FEED_PATH, provider_payload)

    report.update(
        {
            "synced": bool(markets) or (sports_seen > 0 and not warnings),
            "sports_seen": sports_seen,
            "events_seen": events_seen,
            "markets_written": len(markets),
            "target_books_used": sorted(target_books_used),
            "warnings": warnings[:25],
            "requests_remaining": request_usage["requests_remaining"],
            "requests_used": request_usage["requests_used"],
            "last_request_cost": request_usage["last_request_cost"],
        }
    )
    return report


def load_provider_api_key(provider_cfg: dict[str, Any]) -> str:
    env_name = str(provider_cfg.get("api_key_env", "THE_ODDS_API_KEY")).strip() or "THE_ODDS_API_KEY"
    env_value = os.getenv(env_name, "").strip()
    if env_value:
        return env_value
    if not PROVIDER_CREDENTIALS_PATH.exists():
        return ""
    try:
        credentials = load_json(PROVIDER_CREDENTIALS_PATH)
    except Exception:
        return ""
    if not isinstance(credentials, dict):
        return ""
    for key in ("api_key", "the_odds_api_key", env_name):
        value = str(credentials.get(key, "")).strip()
        if value:
            return value
    return ""


def load_cached_provider_report(day_iso: str, provider_cfg: dict[str, Any]) -> dict[str, Any] | None:
    if not provider_cfg.get("use_cache", True):
        return None
    cache_minutes = int(provider_cfg.get("cache_minutes", 20))
    if cache_minutes <= 0:
        return None
    if not PROVIDER_MARKET_FEED_PATH.exists() or not PROVIDER_SYNC_REPORT_PATH.exists():
        return None
    try:
        feed_payload = load_json(PROVIDER_MARKET_FEED_PATH)
        report_payload = load_json(PROVIDER_SYNC_REPORT_PATH)
    except Exception:
        return None
    if not isinstance(feed_payload, dict) or not isinstance(report_payload, dict):
        return None
    feed_rows = feed_payload.get("markets", [])
    if not isinstance(feed_rows, list):
        return None
    if str(feed_payload.get("requested_date", "")).strip() != day_iso:
        return None
    if str(report_payload.get("requested_date", "")).strip() != day_iso or not report_payload.get("synced"):
        return None
    if not feed_rows and report_payload.get("warnings"):
        return None
    generated_at = str(feed_payload.get("generated_at", "")).strip() or str(report_payload.get("generated_at", "")).strip()
    if not generated_at:
        return None
    try:
        generated_dt = datetime.fromisoformat(generated_at)
    except ValueError:
        return None
    age_seconds = (datetime.now() - generated_dt).total_seconds()
    if age_seconds > (cache_minutes * 60):
        return None
    cached = dict(report_payload)
    warnings = list(cached.get("warnings", []))
    warnings.insert(0, f"Used cached provider feed generated at {generated_dt.strftime('%Y-%m-%d %I:%M %p')}.")
    cached["warnings"] = warnings[:25]
    return cached


def load_preserved_provider_report(
    day_iso: str,
    reason: str,
    *,
    provider_name: str = "",
    enabled: bool = True,
) -> dict[str, Any] | None:
    if not PROVIDER_MARKET_FEED_PATH.exists():
        return None
    try:
        feed_payload = load_json(PROVIDER_MARKET_FEED_PATH)
    except Exception:
        return None
    if not isinstance(feed_payload, dict):
        return None
    if str(feed_payload.get("requested_date", "")).strip() != day_iso:
        return None
    feed_rows = feed_payload.get("markets", [])
    if not isinstance(feed_rows, list) or not feed_rows:
        return None

    report_payload: dict[str, Any] = {}
    if PROVIDER_SYNC_REPORT_PATH.exists():
        try:
            loaded_report = load_json(PROVIDER_SYNC_REPORT_PATH)
            if isinstance(loaded_report, dict):
                report_payload = loaded_report
        except Exception:
            report_payload = {}

    preserved = dict(PROVIDER_REPORT_EMPTY)
    preserved.update(report_payload)
    preserved.update(
        {
            "provider": provider_name or str(feed_payload.get("provider", "")).strip() or str(report_payload.get("provider", "")).strip(),
            "enabled": enabled,
            "synced": True,
            "requested_date": day_iso,
            "generated_at": str(feed_payload.get("generated_at", "")).strip() or str(report_payload.get("generated_at", "")).strip(),
            "markets_written": len(feed_rows),
        }
    )
    warnings = [reason, f"Using preserved provider feed with {len(feed_rows)} markets."]
    warnings.extend(str(item) for item in preserved.get("warnings", []) if str(item).strip())
    preserved["warnings"] = warnings[:25]
    return preserved


def odds_api_sport_is_supported(sport: dict[str, Any], requested_markets: list[str]) -> bool:
    key = str(sport.get("key", "")).strip().lower()
    title = str(sport.get("title", "")).strip().lower()
    description = str(sport.get("description", "")).strip().lower()
    if not key:
        return False
    if any(token in key for token in ("winner", "outright", "season_wins")):
        return False
    if "winner" in title or "winner" in description:
        return False
    if not requested_markets:
        return False
    return True


def provider_requested_bookmakers(provider_cfg: dict[str, Any]) -> list[str]:
    ordered: list[str] = []
    for key_name in ("target_book_priority", "reference_bookmakers", "sharp_bookmakers"):
        for raw in provider_cfg.get(key_name, []):
            value = str(raw).strip()
            if value and value not in ordered:
                ordered.append(value)
    return ordered


def build_provider_market_groups(
    *,
    requested_markets: list[str],
    region_count: int,
    remaining_credits: int | None,
    remaining_sports: int,
) -> list[list[str]]:
    markets = [item for item in requested_markets if item]
    if not markets:
        return []

    priority_order = ["h2h", "spreads", "totals"]
    ordered = [item for item in priority_order if item in markets]
    ordered.extend(item for item in markets if item not in ordered)
    full_cost = max(1, len(ordered) * max(1, region_count))
    if remaining_credits is None or remaining_credits < 0 or remaining_credits >= full_cost:
        return [ordered]

    single_market_cost = max(1, region_count)
    if remaining_credits <= (remaining_sports * single_market_cost):
        return [[ordered[0]]] if ordered else []

    budget = max(0, remaining_credits)
    groups: list[list[str]] = []
    for market in ordered:
        if budget < single_market_cost:
            break
        groups.append([market])
        budget -= single_market_cost
    return groups


def build_provider_url(base_url: str, path: str, query: dict[str, Any]) -> str:
    clean_query = {key: value for key, value in query.items() if value not in ("", None)}
    return f"{base_url}{path}?{urllib.parse.urlencode(clean_query)}"


def update_provider_usage(target: dict[str, str], headers: dict[str, Any]) -> None:
    for field, header_name in (
        ("requests_remaining", "x-requests-remaining"),
        ("requests_used", "x-requests-used"),
        ("last_request_cost", "x-requests-last"),
    ):
        value = headers.get(header_name) or headers.get(header_name.title())
        if value not in (None, ""):
            target[field] = str(value)


def provider_sport_sort_key(sport: dict[str, Any], priority_sports: list[str]) -> tuple[int, str, str]:
    sport_key = str(sport.get("key", "")).strip()
    if sport_key in priority_sports:
        return (priority_sports.index(sport_key), str(sport.get("group", "")), str(sport.get("title", "")))
    return (len(priority_sports) + 100, str(sport.get("group", "")), str(sport.get("title", "")))


def fetch_provider_json_with_retry(
    url: str,
    *,
    pause_ms: int = 0,
    retry_count: int = 1,
    retry_pause_ms: int = 1000,
) -> tuple[Any, dict[str, Any]]:
    attempt = 0
    while True:
        if pause_ms > 0:
            time.sleep(pause_ms / 1000.0)
        try:
            return fetch_json_with_headers(url)
        except Exception as exc:
            if "429" not in str(exc) or attempt >= retry_count:
                raise
            attempt += 1
            time.sleep(max(retry_pause_ms, 0) / 1000.0)


def normalize_odds_api_event(
    event: dict[str, Any],
    sport: dict[str, Any],
    requested_markets: list[str],
    provider_cfg: dict[str, Any],
    day_iso: str,
) -> list[dict[str, Any]]:
    bookmakers = build_odds_api_bookmaker_index(event)
    if not bookmakers:
        return []

    sport_key = str(sport.get("key", "")).strip()
    sport_label = normalize_provider_sport_label(sport_key, sport)
    event_id = str(event.get("id", "")).strip()
    home_team = str(event.get("home_team", "")).strip()
    away_team = str(event.get("away_team", "")).strip()
    if not event_id or not home_team or not away_team:
        return []

    event_name = f"{away_team} at {home_team}"
    target_priority = [str(item).strip() for item in provider_cfg.get("target_book_priority", []) if str(item).strip()]
    rows: list[dict[str, Any]] = []

    for market_key in requested_markets:
        target_book_key = select_target_bookmaker(bookmakers, market_key, target_priority)
        if not target_book_key:
            continue
        target_book = bookmakers[target_book_key]
        target_market = target_book["markets"][market_key]

        if market_key == "h2h":
            rows.extend(
                build_provider_h2h_rows(
                    event=event,
                    sport_label=sport_label,
                    sport_key=sport_key,
                    event_name=event_name,
                    event_id=event_id,
                    home_team=home_team,
                    away_team=away_team,
                    bookmakers=bookmakers,
                    target_book_key=target_book_key,
                    target_book=target_book,
                    target_market=target_market,
                    provider_cfg=provider_cfg,
                    day_iso=day_iso,
                )
            )
        elif market_key == "spreads":
            rows.extend(
                build_provider_spread_rows(
                    event=event,
                    sport_label=sport_label,
                    sport_key=sport_key,
                    event_name=event_name,
                    event_id=event_id,
                    home_team=home_team,
                    away_team=away_team,
                    bookmakers=bookmakers,
                    target_book_key=target_book_key,
                    target_book=target_book,
                    target_market=target_market,
                    provider_cfg=provider_cfg,
                    day_iso=day_iso,
                )
            )
        elif market_key == "totals":
            rows.extend(
                build_provider_total_rows(
                    event=event,
                    sport_label=sport_label,
                    sport_key=sport_key,
                    event_name=event_name,
                    event_id=event_id,
                    home_team=home_team,
                    away_team=away_team,
                    bookmakers=bookmakers,
                    target_book_key=target_book_key,
                    target_book=target_book,
                    target_market=target_market,
                    provider_cfg=provider_cfg,
                    day_iso=day_iso,
                )
            )

    return rows


def build_odds_api_bookmaker_index(event: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for bookmaker in event.get("bookmakers", []):
        book_key = str(bookmaker.get("key", "")).strip()
        if not book_key:
            continue
        market_index: dict[str, dict[str, Any]] = {}
        for market in bookmaker.get("markets", []):
            market_key = str(market.get("key", "")).strip()
            if market_key:
                market_index[market_key] = market
        if market_index:
            index[book_key] = {
                "key": book_key,
                "title": str(bookmaker.get("title", "")).strip() or book_key,
                "markets": market_index,
            }
    return index


def select_target_bookmaker(
    bookmakers: dict[str, dict[str, Any]],
    market_key: str,
    target_priority: list[str],
) -> str:
    for book_key in target_priority:
        if book_key in bookmakers and market_key in bookmakers[book_key]["markets"]:
            return book_key
    for book_key, payload in bookmakers.items():
        if market_key in payload["markets"]:
            return book_key
    return ""


def build_provider_h2h_rows(
    *,
    event: dict[str, Any],
    sport_label: str,
    sport_key: str,
    event_name: str,
    event_id: str,
    home_team: str,
    away_team: str,
    bookmakers: dict[str, dict[str, Any]],
    target_book_key: str,
    target_book: dict[str, Any],
    target_market: dict[str, Any],
    provider_cfg: dict[str, Any],
    day_iso: str,
) -> list[dict[str, Any]]:
    prices = h2h_market_prices(target_market, home_team, away_team)
    if not prices:
        return []
    fair_home = devig_probability(prices["home"], prices["away"])
    fair_away = 1.0 - fair_home
    home_consensus = compute_market_consensus_probability(bookmakers, target_book_key, "h2h", "home", None, home_team, away_team, provider_cfg)
    away_consensus = compute_market_consensus_probability(bookmakers, target_book_key, "h2h", "away", None, home_team, away_team, provider_cfg)
    rows: list[dict[str, Any]] = []
    for selection, team, odds, opposite_odds, fair_probability, consensus in (
        ("home", home_team, prices["home"], prices["away"], fair_home, home_consensus),
        ("away", away_team, prices["away"], prices["home"], fair_away, away_consensus),
    ):
        if not consensus or consensus["books"] < int(provider_cfg.get("min_reference_books", 3)):
            continue
        rows.append(
            build_provider_row(
                day_iso=day_iso,
                sport_label=sport_label,
                sport_key=sport_key,
                event_name=event_name,
                event_id=event_id,
                market_type="moneyline",
                bet_type=f"{team} moneyline",
                odds=odds,
                opposite_odds=opposite_odds,
                fair_probability=fair_probability,
                consensus=consensus,
                sportsbook=target_book["title"],
                target_book_key=target_book_key,
                selection=selection,
                selection_team=team,
                home_team=home_team,
                away_team=away_team,
                line="ML",
                notes=f"The Odds API consensus from {consensus['books']} books vs {target_book['title']}.",
            )
        )
    return rows


def build_provider_spread_rows(
    *,
    event: dict[str, Any],
    sport_label: str,
    sport_key: str,
    event_name: str,
    event_id: str,
    home_team: str,
    away_team: str,
    bookmakers: dict[str, dict[str, Any]],
    target_book_key: str,
    target_book: dict[str, Any],
    target_market: dict[str, Any],
    provider_cfg: dict[str, Any],
    day_iso: str,
) -> list[dict[str, Any]]:
    prices = spread_market_prices(target_market, home_team, away_team)
    if not prices:
        return []
    fair_home = devig_probability(prices["home"]["odds"], prices["away"]["odds"])
    fair_away = 1.0 - fair_home
    home_consensus = compute_market_consensus_probability(bookmakers, target_book_key, "spreads", "home", prices["home"]["line"], home_team, away_team, provider_cfg)
    away_consensus = compute_market_consensus_probability(bookmakers, target_book_key, "spreads", "away", prices["away"]["line"], home_team, away_team, provider_cfg)
    rows: list[dict[str, Any]] = []
    for selection, team, side, opposite, fair_probability, consensus in (
        ("home", home_team, prices["home"], prices["away"], fair_home, home_consensus),
        ("away", away_team, prices["away"], prices["home"], fair_away, away_consensus),
    ):
        if not consensus or consensus["books"] < int(provider_cfg.get("min_reference_books", 3)):
            continue
        rows.append(
            build_provider_row(
                day_iso=day_iso,
                sport_label=sport_label,
                sport_key=sport_key,
                event_name=event_name,
                event_id=event_id,
                market_type="spread",
                bet_type=f"{team} {side['line']:+.1f}",
                odds=side["odds"],
                opposite_odds=opposite["odds"],
                fair_probability=fair_probability,
                consensus=consensus,
                sportsbook=target_book["title"],
                target_book_key=target_book_key,
                selection=selection,
                selection_team=team,
                home_team=home_team,
                away_team=away_team,
                line=f"{side['line']:+.1f}",
                notes=f"The Odds API same-line spread consensus from {consensus['books']} books at {side['line']:+.1f}.",
            )
        )
    return rows


def build_provider_total_rows(
    *,
    event: dict[str, Any],
    sport_label: str,
    sport_key: str,
    event_name: str,
    event_id: str,
    home_team: str,
    away_team: str,
    bookmakers: dict[str, dict[str, Any]],
    target_book_key: str,
    target_book: dict[str, Any],
    target_market: dict[str, Any],
    provider_cfg: dict[str, Any],
    day_iso: str,
) -> list[dict[str, Any]]:
    prices = total_market_prices(target_market)
    if not prices:
        return []
    fair_over = devig_probability(prices["over"]["odds"], prices["under"]["odds"])
    fair_under = 1.0 - fair_over
    over_consensus = compute_market_consensus_probability(bookmakers, target_book_key, "totals", "over", prices["over"]["line"], home_team, away_team, provider_cfg)
    under_consensus = compute_market_consensus_probability(bookmakers, target_book_key, "totals", "under", prices["under"]["line"], home_team, away_team, provider_cfg)
    rows: list[dict[str, Any]] = []
    for selection, side, opposite, fair_probability, consensus in (
        ("over", prices["over"], prices["under"], fair_over, over_consensus),
        ("under", prices["under"], prices["over"], fair_under, under_consensus),
    ):
        if not consensus or consensus["books"] < int(provider_cfg.get("min_reference_books", 3)):
            continue
        rows.append(
            build_provider_row(
                day_iso=day_iso,
                sport_label=sport_label,
                sport_key=sport_key,
                event_name=event_name,
                event_id=event_id,
                market_type="total",
                bet_type=f"{selection.title()} {side['line']}",
                odds=side["odds"],
                opposite_odds=opposite["odds"],
                fair_probability=fair_probability,
                consensus=consensus,
                sportsbook=target_book["title"],
                target_book_key=target_book_key,
                selection=selection,
                selection_team="",
                home_team=home_team,
                away_team=away_team,
                line=f"{selection[0].upper()}{side['line']}",
                notes=f"The Odds API same-line total consensus from {consensus['books']} books at {side['line']:.1f}.",
            )
        )
    return rows


def build_provider_row(
    *,
    day_iso: str,
    sport_label: str,
    sport_key: str,
    event_name: str,
    event_id: str,
    market_type: str,
    bet_type: str,
    odds: int,
    opposite_odds: int | None,
    fair_probability: float,
    consensus: dict[str, Any],
    sportsbook: str,
    target_book_key: str,
    selection: str,
    selection_team: str,
    home_team: str,
    away_team: str,
    line: str,
    notes: str,
) -> dict[str, Any]:
    sharp_note = f" Sharp anchor from {consensus['sharp_books']} book(s)." if consensus.get("sharp_books", 0) else ""
    return {
        "date": day_iso,
        "sport": sport_label,
        "event": event_name,
        "event_id": f"oddsapi-{event_id}",
        "market_type": market_type,
        "bet_type": bet_type,
        "odds": format_american(odds),
        "opposite_odds": "" if opposite_odds is None else format_american(opposite_odds),
        "odds_format": "american",
        "fair_probability": round(fair_probability * 100.0, 2),
        "consensus_probability": round(consensus["probability"] * 100.0, 2),
        "sharp_probability": "" if consensus["sharp_probability"] is None else round(consensus["sharp_probability"] * 100.0, 2),
        "quality": round(consensus_quality(consensus), 1),
        "sample_size": consensus_sample_size(consensus),
        "sportsbook": sportsbook,
        "target_bookmaker_key": target_book_key,
        "reference_book_count": consensus["books"],
        "sharp_book_count": consensus["sharp_books"],
        "provider_sport_key": sport_key,
        "line": line,
        "selection": selection,
        "selection_team": selection_team,
        "home_team": home_team,
        "away_team": away_team,
        "notes": f"{notes}{sharp_note}",
    }


def compute_market_consensus_probability(
    bookmakers: dict[str, dict[str, Any]],
    target_book_key: str,
    market_key: str,
    selection: str,
    line_value: float | None,
    home_team: str,
    away_team: str,
    provider_cfg: dict[str, Any],
) -> dict[str, Any] | None:
    reference_books = {str(item).strip() for item in provider_cfg.get("reference_bookmakers", []) if str(item).strip()}
    sharp_books = {str(item).strip() for item in provider_cfg.get("sharp_bookmakers", []) if str(item).strip()}
    sharp_weight = float(provider_cfg.get("sharp_consensus_weight", 0.65))
    weighted_total = 0.0
    total_weight = 0.0
    sharp_total = 0.0
    sharp_weight_total = 0.0
    books = 0
    sharp_count = 0

    for book_key, payload in bookmakers.items():
        if book_key == target_book_key:
            continue
        if reference_books and book_key not in reference_books:
            continue
        market = payload["markets"].get(market_key)
        if not market:
            continue
        probability = extract_market_probability(market_key, market, selection, line_value, home_team, away_team)
        if probability is None:
            continue
        books += 1
        weight = sharp_weight if book_key in sharp_books else 1.0
        weighted_total += probability * weight
        total_weight += weight
        if book_key in sharp_books:
            sharp_total += probability * weight
            sharp_weight_total += weight
            sharp_count += 1

    if books == 0 or total_weight <= 0.0:
        return None

    probability = weighted_total / total_weight
    sharp_probability = (sharp_total / sharp_weight_total) if sharp_weight_total > 0.0 else None
    return {
        "probability": clamp(probability, 0.01, 0.99),
        "sharp_probability": None if sharp_probability is None else clamp(sharp_probability, 0.01, 0.99),
        "books": books,
        "sharp_books": sharp_count,
    }


def h2h_market_prices(market: dict[str, Any], home_team: str, away_team: str) -> dict[str, int] | None:
    outcomes = market.get("outcomes", [])
    home_outcome = next((item for item in outcomes if str(item.get("name", "")).strip() == home_team), None)
    away_outcome = next((item for item in outcomes if str(item.get("name", "")).strip() == away_team), None)
    if not home_outcome or not away_outcome or len(outcomes) != 2:
        return None
    home_price = parse_american(home_outcome.get("price"))
    away_price = parse_american(away_outcome.get("price"))
    if home_price is None or away_price is None:
        return None
    return {"home": home_price, "away": away_price}


def spread_market_prices(market: dict[str, Any], home_team: str, away_team: str) -> dict[str, dict[str, float | int]] | None:
    outcomes = market.get("outcomes", [])
    home_outcome = next((item for item in outcomes if str(item.get("name", "")).strip() == home_team), None)
    away_outcome = next((item for item in outcomes if str(item.get("name", "")).strip() == away_team), None)
    if not home_outcome or not away_outcome:
        return None
    home_price = parse_american(home_outcome.get("price"))
    away_price = parse_american(away_outcome.get("price"))
    home_line = parse_numeric(home_outcome.get("point"))
    away_line = parse_numeric(away_outcome.get("point"))
    if home_price is None or away_price is None or math.isnan(home_line) or math.isnan(away_line):
        return None
    return {
        "home": {"odds": home_price, "line": home_line},
        "away": {"odds": away_price, "line": away_line},
    }


def total_market_prices(market: dict[str, Any]) -> dict[str, dict[str, float | int]] | None:
    outcomes = market.get("outcomes", [])
    over_outcome = next((item for item in outcomes if str(item.get("name", "")).strip().lower() == "over"), None)
    under_outcome = next((item for item in outcomes if str(item.get("name", "")).strip().lower() == "under"), None)
    if not over_outcome or not under_outcome:
        return None
    over_price = parse_american(over_outcome.get("price"))
    under_price = parse_american(under_outcome.get("price"))
    over_line = parse_numeric(over_outcome.get("point"))
    under_line = parse_numeric(under_outcome.get("point"))
    if over_price is None or under_price is None or math.isnan(over_line) or math.isnan(under_line) or not points_match(over_line, under_line):
        return None
    return {
        "over": {"odds": over_price, "line": over_line},
        "under": {"odds": under_price, "line": under_line},
    }


def extract_market_probability(
    market_key: str,
    market: dict[str, Any],
    selection: str,
    line_value: float | None,
    home_team: str,
    away_team: str,
) -> float | None:
    if market_key == "h2h":
        prices = h2h_market_prices(market, home_team, away_team)
        if not prices:
            return None
        fair_home = devig_probability(prices["home"], prices["away"])
        return fair_home if selection == "home" else (1.0 - fair_home if selection == "away" else None)

    if market_key == "spreads":
        prices = spread_market_prices(market, home_team, away_team)
        if not prices or line_value is None:
            return None
        if selection == "home" and points_match(float(prices["home"]["line"]), line_value):
            return devig_probability(int(prices["home"]["odds"]), int(prices["away"]["odds"]))
        if selection == "away" and points_match(float(prices["away"]["line"]), line_value):
            return 1.0 - devig_probability(int(prices["home"]["odds"]), int(prices["away"]["odds"]))
        return None

    if market_key == "totals":
        prices = total_market_prices(market)
        if not prices or line_value is None:
            return None
        if selection == "over" and points_match(float(prices["over"]["line"]), line_value):
            return devig_probability(int(prices["over"]["odds"]), int(prices["under"]["odds"]))
        if selection == "under" and points_match(float(prices["under"]["line"]), line_value):
            return 1.0 - devig_probability(int(prices["over"]["odds"]), int(prices["under"]["odds"]))
        return None

    return None


def odds_api_local_date(raw_commence_time: Any, timezone_name: str) -> str:
    text = str(raw_commence_time or "").strip()
    if not text:
        return ""
    try:
        event_time = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return event_time.astimezone(ZoneInfo(timezone_name)).date().isoformat()
    except Exception:
        return text[:10] if len(text) >= 10 else ""


def normalize_provider_sport_label(sport_key: str, sport: dict[str, Any]) -> str:
    if sport_key in ODDS_API_SPORT_DISPLAY_MAP:
        return ODDS_API_SPORT_DISPLAY_MAP[sport_key]
    group = str(sport.get("group", "")).strip().upper()
    if sport_key.startswith("tennis_"):
        return "TENNIS"
    if sport_key.startswith("rugbyleague_"):
        return "RUGBY_LEAGUE"
    if sport_key.startswith("rugbyunion_"):
        return "RUGBY_UNION"
    if sport_key.startswith("cricket_"):
        return "CRICKET"
    if sport_key.startswith("mma_"):
        return "MMA"
    if sport_key.startswith("boxing_"):
        return "BOXING"
    if sport_key.startswith("golf_"):
        return "GOLF"
    if sport_key.startswith("aussierules_"):
        return "AFL"
    if sport_key.startswith("lacrosse_"):
        return "LACROSSE"
    if sport_key.startswith("soccer_"):
        return "SOCCER"
    if sport_key.startswith("icehockey_"):
        return "HOCKEY"
    if sport_key.startswith("basketball_"):
        return "BASKETBALL"
    if sport_key.startswith("baseball_"):
        return "BASEBALL"
    if sport_key.startswith("americanfootball_"):
        return "FOOTBALL"
    if group:
        return slugify(group).upper()
    return slugify(sport_key).upper() or "OTHER"


def derive_external_model_probability(row: dict[str, Any], model_context: dict[str, Any] | None) -> float | None:
    if not model_context:
        return None

    market_type = str(row.get("market_type", "other")).strip().lower()
    selection = str(row.get("selection", "")).strip().lower()
    selection_team = str(row.get("selection_team", "")).strip()
    line_text = str(row.get("line", "")).strip()
    line_value = parse_numeric(line_text)
    model = model_context["model"]
    home = model_context["home"]
    away = model_context["away"]

    if market_type == "moneyline":
        if selection == "home" or selection_team == home["team"]:
            return clamp(model["home_win_prob"], 0.01, 0.99)
        if selection == "away" or selection_team == away["team"]:
            return clamp(1.0 - model["home_win_prob"], 0.01, 0.99)
        return None

    if market_type == "spread" and not math.isnan(line_value) and "spread_sd" in model:
        if selection == "home" or selection_team == home["team"]:
            return clamp(spread_cover_probability("home", line_value, model["margin"], model["spread_sd"]), 0.01, 0.99)
        if selection == "away" or selection_team == away["team"]:
            return clamp(spread_cover_probability("away", line_value, model["margin"], model["spread_sd"]), 0.01, 0.99)
        return None

    if market_type == "total" and not math.isnan(line_value) and "expected_total" in model and "total_sd" in model:
        if selection == "over":
            return clamp(1.0 - normal_cdf(line_value, model["expected_total"], model["total_sd"]), 0.01, 0.99)
        if selection == "under":
            over_prob = 1.0 - normal_cdf(line_value, model["expected_total"], model["total_sd"])
            return clamp(1.0 - over_prob, 0.01, 0.99)
        return None

    return None


def build_model(
    sport: str,
    event: dict[str, Any],
    home: dict[str, Any],
    away: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, float] | None:
    sport_cfg = config["sports"][sport]
    ml_probability = None
    
    # Calculate momentum factor (recent performance trend) - SCALED DOWN for early season
    sample_size = min(home["games"], away["games"])
    sample_confidence = min(1.0, sample_size / 40.0)  # Full confidence at 40+ games
    
    home_momentum = (home["win_pct"] - 0.5) * 2.0 + (home["home_pct"] - 0.5) * 0.5
    away_momentum = (away["win_pct"] - 0.5) * 2.0 + (away["road_pct"] - 0.5) * 0.5
    momentum_factor = (home_momentum - away_momentum) * 0.3 * sample_confidence
    
    # Calculate trend analysis factor (recent form) - SCALED by sample confidence
    home_trend = (home.get("last_10_win_pct", home["win_pct"]) - 0.5) * 1.5 + (home.get("last_5_win_pct", home["win_pct"]) - 0.5) * 1.0
    away_trend = (away.get("last_10_win_pct", away["win_pct"]) - 0.5) * 1.5 + (away.get("last_5_win_pct", away["win_pct"]) - 0.5) * 1.0
    trend_factor = (home_trend - away_trend) * 0.25 * sample_confidence
    
    # Calculate head-to-head historical factor
    home_h2h_advantage = (home.get("h2h_home_win_pct", 0.5) - 0.5) * 0.15
    home_h2h_recent_advantage = (home.get("h2h_recent_win_pct", 0.5) - 0.5) * 0.20
    h2h_factor = (home_h2h_advantage + home_h2h_recent_advantage) * sample_confidence
    
    # Schedule, injury, and weather factors removed - ESPN doesn't provide this data
    schedule_factor = 0.0
    injury_factor = 0.0
    weather_factor = 0.0
    
    # Combined recent performance factor
    recent_performance_factor = momentum_factor + trend_factor
    
    if sport in BASEBALL_SPORTS:
        home_strength = (
            3.0 * (home["win_pct"] - 0.5)  # Reduced from 4.0
            + 0.6 * (home["runs_pg"] - sport_cfg["offense_baseline"])  # Reduced from 0.75
            - 0.35 * (home["team_era"] - sport_cfg["era_baseline"])  # Reduced from 0.45
            - 0.22 * (home["starter_era"] - sport_cfg["era_baseline"])  # Reduced from 0.28
        )
        away_strength = (
            3.0 * (away["win_pct"] - 0.5)
            + 0.6 * (away["runs_pg"] - sport_cfg["offense_baseline"])
            - 0.35 * (away["team_era"] - sport_cfg["era_baseline"])
            - 0.22 * (away["starter_era"] - sport_cfg["era_baseline"])
        )
        home_edge = sport_cfg["home_field"] + 0.6 * (home["home_pct"] - away["road_pct"])  # Reduced from 0.75
        margin = (home_strength - away_strength) + home_edge + recent_performance_factor + schedule_factor + injury_factor + h2h_factor + weather_factor
        
        # Weather affects totals (wind blowing out increases totals, rain decreases)
        wind_out_factor = 0.3 if max(home.get("wind_speed", 0.0), away.get("wind_speed", 0.0)) > 15 else 0.0
        rain_factor = -0.5 if max(home.get("precipitation", 0.0), away.get("precipitation", 0.0)) > 0.5 else 0.0
        
        expected_total = (
            8.4
            + 0.75 * ((home["runs_pg"] - sport_cfg["offense_baseline"]) + (away["runs_pg"] - sport_cfg["offense_baseline"]))
            + 0.28 * ((home["team_era"] - sport_cfg["era_baseline"]) + (away["team_era"] - sport_cfg["era_baseline"]))
            + 0.18 * ((home["starter_era"] - sport_cfg["era_baseline"]) + (away["starter_era"] - sport_cfg["era_baseline"]))
            + wind_out_factor + rain_factor
        )
        
        # Calculate heuristic probability with REGRESSION TO MEAN
        raw_heuristic_prob = logistic(margin)
        # Strong regression to 0.50 for early season (less data = more regression)
        regression_weight = max(0.35, 1.0 - sample_confidence * 0.5)  # 35-60% regression
        heuristic_prob = (raw_heuristic_prob * (1.0 - regression_weight)) + (0.50 * regression_weight)
        
        # CAP extreme probabilities at realistic levels
        # MLB: max 75% for favorites, min 25% for underdogs
        max_prob = 0.75 if sample_confidence > 0.5 else 0.70  # Lower cap early season
        min_prob = 0.25 if sample_confidence > 0.5 else 0.30
        heuristic_prob = clamp(heuristic_prob, min_prob, max_prob)
        
        # Ensemble probability: combine heuristic model with ML prediction
        if ml_probability is not None:
            # Weight ensemble: 50% ML, 50% heuristic when ML available (less weight on ML until proven)
            raw_ensemble = (ml_probability * 0.5) + (heuristic_prob * 0.5)
            # Apply same caps to ensemble
            ensemble_home_win_prob = clamp(raw_ensemble, min_prob, max_prob)
        else:
            ensemble_home_win_prob = heuristic_prob
        
        return {
            "home_win_prob": ensemble_home_win_prob,
            "expected_total": expected_total,
            "margin": margin,
            "spread_sd": 1.85,
            "total_sd": sport_cfg["total_sd"],
            "quality": quality_score(sport, home, away, ["runs_pg", "team_era", "starter_era"], home.get("stats_defaulted", False) or away.get("stats_defaulted", False)),
            "trend_factor": trend_factor,
            "recent_performance_factor": recent_performance_factor,
            "schedule_factor": schedule_factor,
            "injury_factor": injury_factor,
            "h2h_factor": h2h_factor,
            "weather_factor": weather_factor,
            "ml_probability": ml_probability,
            "heuristic_probability": heuristic_prob,
        }

    if sport in BASKETBALL_SPORTS:
        home_strength = (
            2.8 * (home["win_pct"] - 0.5)  # Reduced from 3.7
            + 0.025 * (home["avg_points"] - sport_cfg["points_baseline"])  # Reduced from 0.032
            + 0.008 * (home["fg_pct"] - 46.0)  # Reduced from 0.010
            + 0.007 * (home["three_pct"] - 35.0)  # Reduced from 0.009
            + 0.008 * (home["avg_assists"] - 25.0)  # Reduced from 0.010
            + 0.005 * (home["avg_rebounds"] - 44.0)  # Reduced from 0.006
        )
        away_strength = (
            2.8 * (away["win_pct"] - 0.5)
            + 0.025 * (away["avg_points"] - sport_cfg["points_baseline"])
            + 0.008 * (away["fg_pct"] - 46.0)
            + 0.007 * (away["three_pct"] - 35.0)
            + 0.008 * (away["avg_assists"] - 25.0)
            + 0.005 * (away["avg_rebounds"] - 44.0)
        )
        expected_margin = sport_cfg["home_field"] + 6.0 * (home_strength - away_strength) + 1.2 * (home["home_pct"] - away["road_pct"]) + recent_performance_factor + schedule_factor + injury_factor + h2h_factor
        expected_total = (
            225.0
            + 0.55 * ((home["avg_points"] - sport_cfg["points_baseline"]) + (away["avg_points"] - sport_cfg["points_baseline"]))
            + 0.040 * ((home["fga"] - 88.0) + (away["fga"] - 88.0))
        )
        # Calculate heuristic probability with REGRESSION TO MEAN
        raw_heuristic_prob = 1.0 - normal_cdf(0.0, expected_margin, sport_cfg["spread_sd"])
        # NBA: regression weight based on sample confidence
        regression_weight = max(0.30, 1.0 - sample_confidence * 0.6)  # 30-70% regression
        heuristic_prob = (raw_heuristic_prob * (1.0 - regression_weight)) + (0.50 * regression_weight)
        
        # CAP extreme probabilities: NBA max 72%, min 28%
        max_prob = 0.72 if sample_confidence > 0.5 else 0.68
        min_prob = 0.28 if sample_confidence > 0.5 else 0.32
        heuristic_prob = clamp(heuristic_prob, min_prob, max_prob)
        
        # Ensemble probability: combine heuristic model with ML prediction
        if ml_probability is not None:
            raw_ensemble = (ml_probability * 0.5) + (heuristic_prob * 0.5)
            ensemble_home_win_prob = clamp(raw_ensemble, min_prob, max_prob)
        else:
            ensemble_home_win_prob = heuristic_prob
        
        return {
            "home_win_prob": ensemble_home_win_prob,
            "expected_total": expected_total,
            "margin": expected_margin,
            "spread_sd": sport_cfg["spread_sd"],
            "total_sd": sport_cfg["total_sd"],
            "quality": quality_score(sport, home, away, ["avg_points", "fg_pct", "three_pct", "avg_assists"], home.get("stats_defaulted", False) or away.get("stats_defaulted", False)),
            "trend_factor": trend_factor,
            "recent_performance_factor": recent_performance_factor,
            "schedule_factor": schedule_factor,
            "injury_factor": injury_factor,
            "h2h_factor": h2h_factor,
            "ml_probability": ml_probability,
            "heuristic_probability": heuristic_prob,
        }

    if sport in HOCKEY_SPORTS:
        home_strength = (
            3.2 * (home["points_pct"] - 0.5)  # Reduced from 4.2
            + 0.65 * (home["goals_pg"] - sport_cfg["goals_baseline"])  # Reduced from 0.8
            + 4.5 * (home["save_pct"] - sport_cfg["save_pct_baseline"])  # Reduced from 6.0
        )
        away_strength = (
            3.2 * (away["points_pct"] - 0.5)
            + 0.65 * (away["goals_pg"] - sport_cfg["goals_baseline"])
            + 4.5 * (away["save_pct"] - sport_cfg["save_pct_baseline"])
        )
        expected_margin = sport_cfg["home_field"] + 1.5 * (home_strength - away_strength) + 0.5 * (home["home_pct"] - away["road_pct"]) + recent_performance_factor + schedule_factor + injury_factor + h2h_factor
        expected_total = (
            6.0
            + 0.55 * ((home["goals_pg"] - sport_cfg["goals_baseline"]) + (away["goals_pg"] - sport_cfg["goals_baseline"]))
            - 10.0 * ((home["save_pct"] - sport_cfg["save_pct_baseline"]) + (away["save_pct"] - sport_cfg["save_pct_baseline"]))
        )
        # Calculate heuristic probability with REGRESSION TO MEAN
        raw_heuristic_prob = 1.0 - normal_cdf(0.0, expected_margin, sport_cfg["spread_sd"])
        regression_weight = max(0.32, 1.0 - sample_confidence * 0.55)  # 32-67% regression
        heuristic_prob = (raw_heuristic_prob * (1.0 - regression_weight)) + (0.50 * regression_weight)
        
        # CAP extreme probabilities: NHL max 73%, min 27%
        max_prob = 0.73 if sample_confidence > 0.5 else 0.69
        min_prob = 0.27 if sample_confidence > 0.5 else 0.31
        heuristic_prob = clamp(heuristic_prob, min_prob, max_prob)
        
        # Ensemble probability
        if ml_probability is not None:
            raw_ensemble = (ml_probability * 0.5) + (heuristic_prob * 0.5)
            ensemble_home_win_prob = clamp(raw_ensemble, min_prob, max_prob)
        else:
            ensemble_home_win_prob = heuristic_prob
        
        return {
            "home_win_prob": ensemble_home_win_prob,
            "expected_total": expected_total,
            "margin": expected_margin,
            "spread_sd": sport_cfg["spread_sd"],
            "total_sd": sport_cfg["total_sd"],
            "quality": quality_score(sport, home, away, ["goals_pg", "save_pct", "points_pct"], home.get("stats_defaulted", False) or away.get("stats_defaulted", False)),
            "trend_factor": trend_factor,
            "recent_performance_factor": recent_performance_factor,
            "schedule_factor": schedule_factor,
            "injury_factor": injury_factor,
            "h2h_factor": h2h_factor,
            "ml_probability": ml_probability,
            "heuristic_probability": heuristic_prob,
        }

    if sport in FOOTBALL_SPORTS:
        expected_margin = (
            sport_cfg["home_field"]
            + sport_cfg["record_weight"] * (home["win_pct"] - away["win_pct"]) * 0.85  # Reduce record weight impact
            + sport_cfg["venue_weight"] * (home["home_pct"] - away["road_pct"]) * 0.85
            + recent_performance_factor + schedule_factor + injury_factor + h2h_factor + weather_factor
        )
        expected_total = sport_cfg["total_baseline"] + 3.0 * ((home["win_pct"] - 0.5) + (away["win_pct"] - 0.5))  # Reduced from 4.0
        # Calculate heuristic probability with REGRESSION TO MEAN
        raw_heuristic_prob = 1.0 - normal_cdf(0.0, expected_margin, sport_cfg["spread_sd"])
        # NFL has more variance, higher regression weight
        regression_weight = max(0.35, 1.0 - sample_confidence * 0.5)  # 35-65% regression
        heuristic_prob = (raw_heuristic_prob * (1.0 - regression_weight)) + (0.50 * regression_weight)
        
        # CAP extreme probabilities: NFL max 70%, min 30% (most unpredictable sport)
        max_prob = 0.70 if sample_confidence > 0.5 else 0.65
        min_prob = 0.30 if sample_confidence > 0.5 else 0.35
        heuristic_prob = clamp(heuristic_prob, min_prob, max_prob)
        
        # Ensemble probability
        if ml_probability is not None:
            raw_ensemble = (ml_probability * 0.5) + (heuristic_prob * 0.5)
            ensemble_home_win_prob = clamp(raw_ensemble, min_prob, max_prob)
        else:
            ensemble_home_win_prob = heuristic_prob
        
        return {
            "home_win_prob": ensemble_home_win_prob,
            "expected_total": expected_total,
            "margin": expected_margin,
            "spread_sd": sport_cfg["spread_sd"],
            "total_sd": sport_cfg["total_sd"],
            "quality": quality_score(sport, home, away, [], home.get("stats_defaulted", False) or away.get("stats_defaulted", False)),
            "trend_factor": trend_factor,
            "recent_performance_factor": recent_performance_factor,
            "schedule_factor": schedule_factor,
            "injury_factor": injury_factor,
            "h2h_factor": h2h_factor,
            "weather_factor": weather_factor,
            "ml_probability": ml_probability,
            "heuristic_probability": heuristic_prob,
        }

    if sport in SOCCER_SPORTS:
        home_strength = (
            2.6 * (home["points_pct"] - 0.5)  # Reduced from 3.4
            + 0.6 * (home["goals_pg"] - sport_cfg["goals_baseline"])  # Reduced from 0.8
            + 0.08 * (home["shots_on_target_pg"] - sport_cfg["shots_baseline"])  # Reduced from 0.10
            + 0.008 * (home["possession_pct"] - sport_cfg["possession_baseline"])  # Reduced from 0.010
        )
        away_strength = (
            2.6 * (away["points_pct"] - 0.5)
            + 0.6 * (away["goals_pg"] - sport_cfg["goals_baseline"])
            + 0.08 * (away["shots_on_target_pg"] - sport_cfg["shots_baseline"])
            + 0.008 * (away["possession_pct"] - sport_cfg["possession_baseline"])
        )
        expected_margin = sport_cfg["home_field"] + 0.75 * (home_strength - away_strength) + 0.40 * (home["home_pct"] - away["road_pct"]) + recent_performance_factor + schedule_factor + injury_factor + h2h_factor + weather_factor
        expected_total = (
            2.55
            + 0.60 * ((home["goals_pg"] - sport_cfg["goals_baseline"]) + (away["goals_pg"] - sport_cfg["goals_baseline"]))
            + 0.08 * ((home["shots_on_target_pg"] - sport_cfg["shots_baseline"]) + (away["shots_on_target_pg"] - sport_cfg["shots_baseline"]))
        )
        # Calculate heuristic probability with REGRESSION TO MEAN
        raw_heuristic_prob = 1.0 - normal_cdf(0.0, expected_margin, sport_cfg["spread_sd"])
        regression_weight = max(0.30, 1.0 - sample_confidence * 0.6)  # 30-70% regression
        heuristic_prob = (raw_heuristic_prob * (1.0 - regression_weight)) + (0.50 * regression_weight)
        
        # CAP extreme probabilities: Soccer max 70% (draws make it less predictable), min 30%
        max_prob = 0.70 if sample_confidence > 0.5 else 0.65
        min_prob = 0.30 if sample_confidence > 0.5 else 0.35
        heuristic_prob = clamp(heuristic_prob, min_prob, max_prob)
        
        # Ensemble probability
        if ml_probability is not None:
            raw_ensemble = (ml_probability * 0.5) + (heuristic_prob * 0.5)
            ensemble_home_win_prob = clamp(raw_ensemble, min_prob, max_prob)
        else:
            ensemble_home_win_prob = heuristic_prob
        
        return {
            "home_win_prob": ensemble_home_win_prob,
            "expected_total": expected_total,
            "margin": expected_margin,
            "spread_sd": sport_cfg["spread_sd"],
            "total_sd": sport_cfg["total_sd"],
            "quality": quality_score(sport, home, away, ["goals_pg", "shots_on_target_pg", "possession_pct"], home.get("stats_defaulted", False) or away.get("stats_defaulted", False)),
            "trend_factor": trend_factor,
            "recent_performance_factor": recent_performance_factor,
            "schedule_factor": schedule_factor,
            "injury_factor": injury_factor,
            "h2h_factor": h2h_factor,
            "weather_factor": weather_factor,
            "ml_probability": ml_probability,
            "heuristic_probability": heuristic_prob,
        }

    return None


def build_market_candidates(
    *,
    sport: str,
    event_name: str,
    event_id: str,
    event_date: str,
    odds_block: dict[str, Any],
    model: dict[str, float],
    sample_size: int,
    config: dict[str, Any],
    drawdown_multiplier: float,
    current_bankroll: float,
    history_profile: dict[str, Any],
    book: str,
    home: dict[str, Any],
    away: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    correlation_group = f"{sport}:{matchup_key(home['team'], away['team']) or event_id}"

    moneyline = odds_block.get("moneyline", {})
    total = odds_block.get("total", {})
    point_spread = odds_block.get("pointSpread", {})

    ml_home_odds = extract_odds(moneyline, "home")
    ml_away_odds = extract_odds(moneyline, "away")
    if ml_home_odds is not None and ml_away_odds is not None:
        fair_home = devig_probability(ml_home_odds, ml_away_odds)
        fair_away = 1.0 - fair_home
        home_notes = f"Auto model leans {home['team']} on win rate, venue split, and core team strength."
        away_notes = f"Auto model leans {away['team']} on win rate, venue split, and core team strength."
        candidates.append(
            evaluate_market_candidate(
                event_date=event_date,
                sport=sport,
                event_name=event_name,
                event_id=event_id,
                market_type="moneyline",
                bet_type=f"{home['team']} moneyline",
                odds=ml_home_odds,
                opposite_odds=ml_away_odds,
                fair_probability=fair_home,
                model_probability=model["home_win_prob"],
                quality=model["quality"],
                sample_size=sample_size,
                config=config,
                drawdown_multiplier=drawdown_multiplier,
                current_bankroll=current_bankroll,
                history_profile=history_profile,
                book=book,
                line="ML",
                notes=home_notes,
                correlation_group=correlation_group,
                home_team=home["team"],
                away_team=away["team"],
                selection="home",
                selection_team=home["team"],
                training_features=build_candidate_feature_vector(
                    sport=sport,
                    market_type="moneyline",
                    sample_size=sample_size,
                    quality=model["quality"],
                    fair_probability=fair_home,
                    model_probability=model["home_win_prob"],
                    odds=ml_home_odds,
                    opposite_odds=ml_away_odds,
                    line_value=None,
                    home=home,
                    away=away,
                    model=model,
                ),
            )
        )
        candidates.append(
            evaluate_market_candidate(
                event_date=event_date,
                sport=sport,
                event_name=event_name,
                event_id=event_id,
                market_type="moneyline",
                bet_type=f"{away['team']} moneyline",
                odds=ml_away_odds,
                opposite_odds=ml_home_odds,
                fair_probability=fair_away,
                model_probability=1.0 - model["home_win_prob"],
                quality=model["quality"],
                sample_size=sample_size,
                config=config,
                drawdown_multiplier=drawdown_multiplier,
                current_bankroll=current_bankroll,
                history_profile=history_profile,
                book=book,
                line="ML",
                notes=away_notes,
                correlation_group=correlation_group,
                home_team=home["team"],
                away_team=away["team"],
                selection="away",
                selection_team=away["team"],
                training_features=build_candidate_feature_vector(
                    sport=sport,
                    market_type="moneyline",
                    sample_size=sample_size,
                    quality=model["quality"],
                    fair_probability=fair_away,
                    model_probability=1.0 - model["home_win_prob"],
                    odds=ml_away_odds,
                    opposite_odds=ml_home_odds,
                    line_value=None,
                    home=home,
                    away=away,
                    model=model,
                ),
            )
        )

    if total:
        over_data = extract_total_side(total, "over")
        under_data = extract_total_side(total, "under")
        if over_data and under_data and over_data["line"] is not None:
            over_prob = 1.0 - normal_cdf(over_data["line"], model["expected_total"], model["total_sd"])
            fair_over = devig_probability(over_data["odds"], under_data["odds"])
            fair_under = 1.0 - fair_over
            total_note = f"Auto total projection: {model['expected_total']:.2f} vs market {over_data['line']:.1f}."
            candidates.append(
                evaluate_market_candidate(
                    event_date=event_date,
                    sport=sport,
                    event_name=event_name,
                    event_id=event_id,
                    market_type="total",
                    bet_type=f"Over {over_data['line']}",
                    odds=over_data["odds"],
                    opposite_odds=under_data["odds"],
                    fair_probability=fair_over,
                    model_probability=over_prob,
                    quality=model["quality"] - 3,
                    sample_size=sample_size,
                    config=config,
                    drawdown_multiplier=drawdown_multiplier,
                    current_bankroll=current_bankroll,
                    history_profile=history_profile,
                    book=book,
                    line=f"O{over_data['line']}",
                    notes=total_note,
                    correlation_group=correlation_group,
                    home_team=home["team"],
                    away_team=away["team"],
                    selection="over",
                    selection_team="",
                    training_features=build_candidate_feature_vector(
                        sport=sport,
                        market_type="total",
                        sample_size=sample_size,
                        quality=model["quality"] - 3,
                        fair_probability=fair_over,
                        model_probability=over_prob,
                        odds=over_data["odds"],
                        opposite_odds=under_data["odds"],
                        line_value=over_data["line"],
                        home=home,
                        away=away,
                        model=model,
                    ),
                )
            )
            candidates.append(
                evaluate_market_candidate(
                    event_date=event_date,
                    sport=sport,
                    event_name=event_name,
                    event_id=event_id,
                    market_type="total",
                    bet_type=f"Under {under_data['line']}",
                    odds=under_data["odds"],
                    opposite_odds=over_data["odds"],
                    fair_probability=fair_under,
                    model_probability=1.0 - over_prob,
                    quality=model["quality"] - 3,
                    sample_size=sample_size,
                    config=config,
                    drawdown_multiplier=drawdown_multiplier,
                    current_bankroll=current_bankroll,
                    history_profile=history_profile,
                    book=book,
                    line=f"U{under_data['line']}",
                    notes=total_note,
                    correlation_group=correlation_group,
                    home_team=home["team"],
                    away_team=away["team"],
                    selection="under",
                    selection_team="",
                    training_features=build_candidate_feature_vector(
                        sport=sport,
                        market_type="total",
                        sample_size=sample_size,
                        quality=model["quality"] - 3,
                        fair_probability=fair_under,
                        model_probability=1.0 - over_prob,
                        odds=under_data["odds"],
                        opposite_odds=over_data["odds"],
                        line_value=under_data["line"],
                        home=home,
                        away=away,
                        model=model,
                    ),
                )
            )

    if sport in SPREAD_MODEL_SPORTS and point_spread:
        home_spread = extract_spread_side(point_spread, "home")
        away_spread = extract_spread_side(point_spread, "away")
        if home_spread and away_spread and home_spread["line"] is not None:
            home_cover = spread_cover_probability("home", home_spread["line"], model["margin"], model["spread_sd"])
            fair_home = devig_probability(home_spread["odds"], away_spread["odds"])
            fair_away = 1.0 - fair_home
            spread_note = f"Auto projected margin: {model['margin']:.2f} vs market {home_spread['line']:+.1f}."
            candidates.append(
                evaluate_market_candidate(
                    event_date=event_date,
                    sport=sport,
                    event_name=event_name,
                    event_id=event_id,
                    market_type="spread",
                    bet_type=f"{home['team']} {home_spread['line']:+.1f}",
                    odds=home_spread["odds"],
                    opposite_odds=away_spread["odds"],
                    fair_probability=fair_home,
                    model_probability=home_cover,
                    quality=model["quality"] - 2,
                    sample_size=sample_size,
                    config=config,
                    drawdown_multiplier=drawdown_multiplier,
                    current_bankroll=current_bankroll,
                    history_profile=history_profile,
                    book=book,
                    line=f"{home_spread['line']:+.1f}",
                    notes=spread_note,
                    correlation_group=correlation_group,
                    home_team=home["team"],
                    away_team=away["team"],
                    selection="home",
                    selection_team=home["team"],
                    training_features=build_candidate_feature_vector(
                        sport=sport,
                        market_type="spread",
                        sample_size=sample_size,
                        quality=model["quality"] - 2,
                        fair_probability=fair_home,
                        model_probability=home_cover,
                        odds=home_spread["odds"],
                        opposite_odds=away_spread["odds"],
                        line_value=home_spread["line"],
                        home=home,
                        away=away,
                        model=model,
                    ),
                )
            )
            candidates.append(
                evaluate_market_candidate(
                    event_date=event_date,
                    sport=sport,
                    event_name=event_name,
                    event_id=event_id,
                    market_type="spread",
                    bet_type=f"{away['team']} {away_spread['line']:+.1f}",
                    odds=away_spread["odds"],
                    opposite_odds=home_spread["odds"],
                    fair_probability=fair_away,
                    model_probability=spread_cover_probability("away", away_spread["line"], model["margin"], model["spread_sd"]),
                    quality=model["quality"] - 2,
                    sample_size=sample_size,
                    config=config,
                    drawdown_multiplier=drawdown_multiplier,
                    current_bankroll=current_bankroll,
                    history_profile=history_profile,
                    book=book,
                    line=f"{away_spread['line']:+.1f}",
                    notes=spread_note,
                    correlation_group=correlation_group,
                    home_team=home["team"],
                    away_team=away["team"],
                    selection="away",
                    selection_team=away["team"],
                    training_features=build_candidate_feature_vector(
                        sport=sport,
                        market_type="spread",
                        sample_size=sample_size,
                        quality=model["quality"] - 2,
                        fair_probability=fair_away,
                        model_probability=spread_cover_probability("away", away_spread["line"], model["margin"], model["spread_sd"]),
                        odds=away_spread["odds"],
                        opposite_odds=home_spread["odds"],
                        line_value=away_spread["line"],
                        home=home,
                        away=away,
                        model=model,
                    ),
                )
            )

    return [candidate for candidate in candidates if candidate]


def evaluate_market_candidate(
    *,
    event_date: str,
    sport: str,
    event_name: str,
    event_id: str,
    market_type: str,
    bet_type: str,
    odds: int,
    opposite_odds: int | None,
    fair_probability: float,
    model_probability: float,
    quality: float,
    sample_size: int,
    config: dict[str, Any],
    drawdown_multiplier: float,
    current_bankroll: float,
    history_profile: dict[str, Any],
    book: str,
    line: str,
    notes: str,
    correlation_group: str | None = None,
    consensus_probability: float | None = None,
    sharp_probability: float | None = None,
    reference_book_count: int = 0,
    sharp_book_count: int = 0,
    home_team: str = "",
    away_team: str = "",
    selection: str = "",
    selection_team: str = "",
    training_features: list[float] | None = None,
) -> dict[str, Any]:
    raw_model_gap_pct = abs(model_probability - fair_probability) * 100.0
    raw_extreme_pct = abs(model_probability - 0.5) * 100.0
    
    # Sport-specific realistic probability caps
    sport_caps = {
        "MLB": (0.25, 0.75),  # Baseball: 25%-75%
        "NCAABASE": (0.25, 0.75),
        "NBA": (0.28, 0.72),  # Basketball: 28%-72%
        "WNBA": (0.28, 0.72),
        "NCAAMB": (0.28, 0.72),
        "NCAAWB": (0.28, 0.72),
        "NHL": (0.27, 0.73),  # Hockey: 27%-73%
        "NFL": (0.30, 0.70),  # Football: 30%-70%
        "NCAAF": (0.30, 0.70),
    }
    min_prob_cap, max_prob_cap = sport_caps.get(sport, (0.28, 0.72))
    
    # Adjust caps based on sample size (early season = more conservative)
    sample_confidence = min(1.0, sample_size / 40.0)
    if sample_confidence < 0.5:
        # Tighten caps for early season (less data = less confidence)
        min_prob_cap = min_prob_cap + 0.05
        max_prob_cap = max_prob_cap - 0.05
    
    # Cap model probability at realistic sport-specific limits
    capped_model_probability = clamp(model_probability, min_prob_cap, max_prob_cap)
    
    # Calibrated probability with regression to mean
    calibration_weight = 0.85 if sample_confidence > 0.5 else 0.70  # More regression early season
    calibrated_probability = (capped_model_probability * calibration_weight) + (0.50 * (1.0 - calibration_weight))
    reference_probability = sharp_probability if sharp_probability is not None else consensus_probability
    reference_gap_pct = abs(reference_probability - fair_probability) * 100.0 if reference_probability is not None else 0.0
    market_respect_weight = clamp(
        config["market_respect_base"] + (quality * config["market_respect_quality_scale"]),
        0.10,
        0.38,
    )
    probability_gap_cap = config["max_probability_gap_pct"] / 100.0
    provider_supported = False
    if reference_probability is not None:
        aligned_signal = (reference_probability - fair_probability) * (calibrated_probability - fair_probability) > 0
        if aligned_signal and reference_gap_pct >= float(config.get("provider_reference_gap_min_pct", 1.5)):
            extra_capture = float(config.get("provider_reference_signal_capture_pct", 14.0)) / 100.0
            extra_capture += min(0.06, max(0, reference_book_count - 3) * 0.01)
            extra_capture += min(0.04, max(0, sharp_book_count) * 0.01)
            extra_capture = clamp(extra_capture, 0.08, 0.24)
            if reference_probability >= fair_probability:
                supported_probability = reference_probability + max(0.0, calibrated_probability - reference_probability) * extra_capture
            else:
                supported_probability = reference_probability + min(0.0, calibrated_probability - reference_probability) * extra_capture
            reference_gap_allowance = max(probability_gap_cap, abs(reference_probability - fair_probability) + 0.03)
            market_respected_probability = clamp(
                supported_probability,
                max(0.01, fair_probability - reference_gap_allowance),
                min(0.99, fair_probability + reference_gap_allowance),
            )
            provider_supported = True
        else:
            anchored_probability = reference_probability + ((calibrated_probability - reference_probability) * 0.10)
            market_respected_probability = fair_probability + ((anchored_probability - fair_probability) * market_respect_weight)
            market_respected_probability = clamp(
                market_respected_probability,
                max(0.01, fair_probability - probability_gap_cap),
                min(0.99, fair_probability + probability_gap_cap),
            )
    else:
        market_respected_probability = fair_probability + ((calibrated_probability - fair_probability) * market_respect_weight)
        market_respected_probability = clamp(
            market_respected_probability,
            max(0.01, fair_probability - probability_gap_cap),
            min(0.99, fair_probability + probability_gap_cap),
        )
    quality_penalty = max(0.0, 75.0 - quality) / 1000.0
    # Increased sample penalty for early season (insufficient data)
    # Normal penalty: 1.5% per game under 18
    # Early season (<20 games): additional 3% penalty
    base_sample_penalty = max(0.0, 20 - sample_size) / 1000.0  # Increased threshold from 18 to 20
    early_season_penalty = max(0.0, (20 - sample_size) * 0.003) if sample_size < 20 else 0.0
    sample_penalty = base_sample_penalty + early_season_penalty
    missing_opposite_penalty = 0.0 if opposite_odds is not None else 0.005
    history_adjustment = history_penalty_for_market(history_profile, sport, market_type)
    gap_penalty_pct = min(
        config["model_gap_penalty_cap_pct"],
        max(0.0, raw_model_gap_pct - config["soft_model_gap_pct"]) * config["model_gap_penalty_scale"],
    )
    extreme_penalty_pct = min(
        config["model_extreme_penalty_cap_pct"],
        max(0.0, raw_extreme_pct - config["soft_model_extreme_pct"]) * config["model_extreme_penalty_scale"],
    )
    uncertainty_haircut = (
        config["base_haircut_pct"] / 100.0
        + MARKET_HAIRCUTS[market_type] / 100.0
        + (config["auto_model_penalty_pct"] / 100.0)
        + quality_penalty
        + sample_penalty
        + missing_opposite_penalty
        + (history_adjustment["penalty_pct"] / 100.0)
        + (gap_penalty_pct / 100.0)
        + (extreme_penalty_pct / 100.0)
    )
    if provider_supported:
        uncertainty_haircut *= float(config.get("provider_supported_haircut_multiplier", 0.72))
    base_probability = clamp(market_respected_probability - uncertainty_haircut, 0.01, 0.99)
    history_confidence_bonus = history_adjustment.get("boost_pct", 0.0) / 100.0
    if history_confidence_bonus > 0.0:
        base_probability = clamp(base_probability + history_confidence_bonus, 0.01, 0.99)
    
    # Apply weighted blend to combine model probability with market probability
    bayesian_weight = float(config.get("true_probability_weight", 0.7))
    true_probability = weighted_blend(model_probability, base_probability, bayesian_weight)
    candidate_ml_probability = ml_manager.predict_market(sport, market_type, training_features)
    ml_blend_weight = 0.0
    if candidate_ml_probability is not None:
        sample_confidence_ml = min(1.0, sample_size / 60.0)
        ml_blend_weight = clamp(0.08 + (0.12 * sample_confidence_ml), 0.08, 0.20)
        true_probability = weighted_blend(true_probability, candidate_ml_probability, ml_blend_weight)
    
    implied_probability = american_to_probability(odds)
    payout_multiple = american_to_profit_multiple(odds)
    ev = (true_probability * payout_multiple) - (1.0 - true_probability)
    edge = true_probability - fair_probability
    
    # Calculate sample confidence for Kelly adjustment
    sample_confidence_kelly = min(1.0, sample_size / 40.0)
    
    # Kelly criterion with uncertainty adjustment
    full_kelly = max(0.0, ((payout_multiple * true_probability) - (1.0 - true_probability)) / payout_multiple)
    
    # Reduce Kelly fraction for low sample sizes (early season = less confidence)
    # At 10 games: 40% of normal kelly fraction
    # At 40+ games: 100% of normal kelly fraction
    sample_kelly_multiplier = 0.4 + (0.6 * sample_confidence_kelly)
    
    # Also apply uncertainty penalty to EV edge for stake calculation
    edge_confidence = sample_confidence_kelly * (quality / 100.0)
    adjusted_kelly_fraction = config["kelly_fraction"] * sample_kelly_multiplier * edge_confidence
    
    stake_pct = min(config["max_stake_pct"], full_kelly * adjusted_kelly_fraction * drawdown_multiplier)
    
    # Hard cap on single bet size: max 3% for early season (<20 games)
    max_single_bet = 0.03 if sample_size < 20 else config["max_stake_pct"]
    stake_pct = min(stake_pct, max_single_bet)
    
    stake = current_bankroll * stake_pct
    reliability = clamp(
        45.0
        + (quality * 0.4)
        + (sample_size * 0.35)
        - (uncertainty_haircut * 100.0 * 2.3),
        0.0,
        100.0,
    )
    qualified = (
        (ev * 100.0) >= config["min_ev_pct"]
        and (edge * 100.0) >= config["min_edge_pct"]
        and quality >= config["quality_floor"]
        and (true_probability * 100.0) >= float(config.get("min_true_probability_pct", 0.0))
        and sample_size >= int(config.get("min_sample_size", 10))
    )
    probability_bonus = true_probability * 100.0 * float(config.get("true_probability_weight", 0.45))
    longshot_penalty = max(0.0, payout_multiple - float(config.get("longshot_penalty_threshold", 1.6))) * float(config.get("longshot_penalty_scale", 5.0))
    unsupported_positive_odds_cap = int(config.get("max_positive_odds_without_provider", 0) or 0)
    public_longshot_penalty = 0.0
    if not provider_supported and unsupported_positive_odds_cap > 0 and odds > unsupported_positive_odds_cap:
        public_longshot_penalty = min(12.0, (odds - unsupported_positive_odds_cap) * 0.08)
    
    # Enhanced composite score calculation with additional factors
    ev_weight = 4.2  # Increased from 3.4
    edge_weight = 3.4  # Increased from 2.6
    reliability_weight = 0.5  # Increased from 0.35
    probability_bonus_weight = 1.2  # New weight for probability bonus
    
    # Add market respect factor
    market_respect_bonus = 0.0
    if market_respected_probability and true_probability:
        gap = abs(market_respected_probability - true_probability) * 100.0
        if gap < config.get("max_probability_gap_pct", 12.0):
            market_respect_bonus = (12.0 - gap) * 0.25
    
    # Add provider support bonus
    provider_bonus = 0.0
    if provider_supported:
        provider_bonus = 8.0
    
    # Add quality bonus for high-quality data
    quality_bonus = 0.0
    if quality >= 75.0:
        quality_bonus = (quality - 75.0) * 0.15
    
    # Add sample size bonus for large sample sizes
    sample_size_bonus = 0.0
    if sample_size >= 50:
        sample_size_bonus = min(5.0, (sample_size - 50) * 0.05)
    history_bonus = float(history_adjustment.get("boost_pct", 0.0)) * float(config.get("history_boost_score_multiplier", 4.5))
    
    # Calculate enhanced composite score
    composite_score = (
        (ev * 100.0 * ev_weight) +
        (edge * 100.0 * edge_weight) +
        (reliability * reliability_weight) +
        (probability_bonus * probability_bonus_weight) +
        market_respect_bonus +
        provider_bonus +
        quality_bonus +
        history_bonus +
        sample_size_bonus -
        longshot_penalty -
        public_longshot_penalty
    )
    filter_reasons: list[str] = []
    if (ev * 100.0) < config["min_ev_pct"]:
        filter_reasons.append("EV below floor")
    if (edge * 100.0) < config["min_edge_pct"]:
        filter_reasons.append("Edge below floor")
    if quality < config["quality_floor"]:
        filter_reasons.append("Quality below floor")
    if (true_probability * 100.0) < float(config.get("min_true_probability_pct", 0.0)):
        filter_reasons.append("True probability below floor")
    if sample_size < int(config.get("min_sample_size", 10)):
        filter_reasons.append(f"Sample size below minimum ({sample_size} < {config.get('min_sample_size', 10)})")
    if not provider_supported and unsupported_positive_odds_cap > 0 and odds > unsupported_positive_odds_cap:
        filter_reasons.append(f"Positive odds above no-provider cap (+{unsupported_positive_odds_cap})")
    qualified = qualified and not (
        not provider_supported and unsupported_positive_odds_cap > 0 and odds > unsupported_positive_odds_cap
    )
    final_notes = notes
    if history_adjustment["summary"]:
        final_notes = f"{notes} History signal: {history_adjustment['summary']}."

    return {
        "id": f"{event_id}-{market_type}-{slugify(bet_type)}",
        "eventId": event_id,
        "date": event_date,
        "sport": sport,
        "event": event_name,
        "betType": bet_type,
        "marketType": market_type,
        "odds": odds,
        "oppositeOdds": opposite_odds,
        "modelProbability": round(capped_model_probability * 100.0, 2),
        "raw_model_gap_pct": round(raw_model_gap_pct, 2),
        "raw_extreme_pct": round(raw_extreme_pct, 2),
        "dataQuality": round(quality, 1),
        "sampleSize": int(sample_size),
        "sportsbook": book,
        "source": "espn",
        "line": line,
        "correlationGroup": correlation_group or event_id,
        "homeTeam": home_team,
        "awayTeam": away_team,
        "selection": selection,
        "selectionTeam": selection_team,
        "notes": final_notes,
        "implied_probability_pct": round(implied_probability * 100.0, 2),
        "fair_probability_pct": round(fair_probability * 100.0, 2),
        "reference_probability_pct": "" if reference_probability is None else round(reference_probability * 100.0, 2),
        "market_respected_probability_pct": round(market_respected_probability * 100.0, 2),
        "true_probability_pct": round(true_probability * 100.0, 2),
        "candidate_ml_probability_pct": "" if candidate_ml_probability is None else round(candidate_ml_probability * 100.0, 2),
        "candidate_ml_blend_weight_pct": round(ml_blend_weight * 100.0, 2),
        "ev_pct": round(ev * 100.0, 2),
        "edge_pct": round(edge * 100.0, 2),
        "stake": round(stake, 2),
        "stake_pct": round(stake_pct * 100.0, 2),
        "bankrollBase": round(current_bankroll, 2),
        "max_odds": probability_to_american(true_probability),
        "edge_score": round(reliability),
        "history_penalty_pct": round(history_adjustment["penalty_pct"], 2),
        "history_boost_pct": round(history_adjustment.get("boost_pct", 0.0), 2),
        "model_gap_penalty_pct": round(gap_penalty_pct, 2),
        "model_extreme_penalty_pct": round(extreme_penalty_pct, 2),
        "probability_bonus": round(probability_bonus, 2),
        "longshot_penalty": round(longshot_penalty, 2),
        "public_longshot_penalty": round(public_longshot_penalty, 2),
        "history_context": history_adjustment["summary"],
        "provider_supported": provider_supported,
        "training_features": training_features or [],
        "filter_reasons": filter_reasons,
        "qualified": qualified,
        "composite_score": round(composite_score, 2),
    }


def calculate_correlation(bet1: dict[str, Any], bet2: dict[str, Any]) -> float:
    """Calculate correlation between two bets based on sport, event, and market type."""
    if bet1["correlationGroup"] == bet2["correlationGroup"]:
        return 1.0  # Same event = fully correlated
    
    correlation_score = 0.0
    
    # Sport correlation
    if bet1["sport"] == bet2["sport"]:
        correlation_score += 0.3
    
    # Market type correlation (same market type on different events)
    if bet1["marketType"] == bet2["marketType"]:
        correlation_score += 0.2
    
    # Same team involved
    bet1_teams = {bet1.get("homeTeam", ""), bet1.get("awayTeam", "")}
    bet2_teams = {bet2.get("homeTeam", ""), bet2.get("awayTeam", "")}
    if bet1_teams & bet2_teams:
        correlation_score += 0.4
    
    return min(correlation_score, 1.0)


def pick_top_candidates(candidates: list[dict[str, Any]], config: dict[str, Any], top_n: int) -> list[dict[str, Any]]:
    picks: list[dict[str, Any]] = []
    sport_counts: dict[str, int] = {}
    market_counts: dict[str, int] = {}
    seen_events: set[str] = set()
    pick_limit = resolve_pick_limit(config, top_n)
    correlation_threshold = config.get("correlation_threshold", 0.7)
    max_correlated_bets = config.get("max_correlated_bets", 1)

    qualified = shortlist_best_bet_candidates(
        [candidate for candidate in candidates if candidate["qualified"]],
        config,
    )
    for enforce_diversity in (True, False):
        for candidate in qualified:
            if pick_limit is not None and len(picks) >= pick_limit:
                return picks
            if candidate["correlationGroup"] in seen_events:
                continue
            if enforce_diversity:
                if sport_counts.get(candidate["sport"], 0) >= config["max_per_sport"]:
                    continue
                if market_counts.get(candidate["marketType"], 0) >= config["max_per_market"]:
                    continue
            
            # Check correlation with existing picks
            correlated_count = 0
            for existing_pick in picks:
                correlation = calculate_correlation(candidate, existing_pick)
                if correlation >= correlation_threshold:
                    correlated_count += 1
                    if correlated_count >= max_correlated_bets:
                        break
            
            if correlated_count < max_correlated_bets:
                picks.append(candidate)
                seen_events.add(candidate["correlationGroup"])
                sport_counts[candidate["sport"]] = sport_counts.get(candidate["sport"], 0) + 1
                market_counts[candidate["marketType"]] = market_counts.get(candidate["marketType"], 0) + 1
    return rebalance_shortlist_stakes(picks, config)


def resolve_pick_limit(config: dict[str, Any], requested_limit: int) -> int | None:
    limits = []
    if requested_limit and requested_limit > 0:
        limits.append(int(requested_limit))
    config_limit = int(config.get("max_total_picks", 0) or 0)
    if config_limit > 0:
        limits.append(config_limit)
    if bool(config.get("_provider_safe_mode")):
        no_provider_limit = int(config.get("max_total_picks_without_provider", 0) or 0)
        if no_provider_limit > 0:
            limits.append(no_provider_limit)
    return min(limits) if limits else None


def shortlist_best_bet_candidates(candidates: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    if not candidates:
        return candidates

    ordered = sorted(candidates, key=lambda item: item["composite_score"], reverse=True)
    best_score = float(ordered[0].get("composite_score", 0.0))
    abs_floor = float(config.get("best_bet_min_composite_score", 140.0))
    relative_ratio = float(config.get("best_bet_score_ratio", 0.62))
    max_score_drop = float(config.get("best_bet_max_score_drop", 140.0))
    min_edge_score = float(config.get("best_bet_min_edge_score", 72.0))
    min_ev = float(config.get("best_bet_min_ev_pct", config.get("min_ev_pct", 0.0)))

    score_floor = max(abs_floor, best_score * relative_ratio, best_score - max_score_drop)
    best_only = [
        candidate
        for candidate in ordered
        if float(candidate.get("composite_score", 0.0)) >= score_floor
        and float(candidate.get("edge_score", 0.0)) >= min_edge_score
        and float(candidate.get("ev_pct", 0.0)) >= min_ev
    ]
    if best_only:
        return best_only

    fallback = [
        candidate
        for candidate in ordered
        if float(candidate.get("edge_score", 0.0)) >= min_edge_score
    ]
    return fallback or ordered


def deduplicate_candidate_sources(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    preferred: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    source_rank = {"provider_feed": 3, "external_feed": 2, "espn": 1}
    for candidate in candidates:
        source = str(candidate.get("source", "espn")).strip().lower() or "espn"
        key = (
            str(candidate.get("sport", "")).strip().upper(),
            slugify(str(candidate.get("event", ""))),
            str(candidate.get("marketType", "")).strip().lower(),
            slugify(str(candidate.get("betType", ""))),
        )
        current = preferred.get(key)
        if current is None:
            preferred[key] = candidate
            continue
        current_source = str(current.get("source", "espn")).strip().lower() or "espn"
        if source_rank.get(source, 0) > source_rank.get(current_source, 0):
            preferred[key] = candidate
    return list(preferred.values())


def build_runtime_config(config: dict[str, Any], provider_report: dict[str, Any]) -> dict[str, Any]:
    effective = copy.deepcopy(config)
    runtime_notes: list[str] = []
    provider_safe = bool(config.get("provider_feed", {}).get("enabled")) and not provider_report.get("synced")
    effective["_provider_safe_mode"] = provider_safe
    if not provider_safe:
        effective["_runtime_notes"] = runtime_notes
        return effective

    if "no_provider_min_ev_pct" in config:
        effective["min_ev_pct"] = max(float(effective.get("min_ev_pct", 0.0)), float(config["no_provider_min_ev_pct"]))
    if "no_provider_min_edge_pct" in config:
        effective["min_edge_pct"] = max(float(effective.get("min_edge_pct", 0.0)), float(config["no_provider_min_edge_pct"]))
    if "no_provider_min_true_probability_pct" in config:
        effective["min_true_probability_pct"] = max(float(effective.get("min_true_probability_pct", 0.0)), float(config["no_provider_min_true_probability_pct"]))
    if "no_provider_quality_floor" in config:
        effective["quality_floor"] = max(int(effective.get("quality_floor", 0)), int(config["no_provider_quality_floor"]))
    if "no_provider_max_daily_risk_pct" in config:
        effective["max_daily_risk_pct"] = min(float(effective.get("max_daily_risk_pct", 0.12)), float(config["no_provider_max_daily_risk_pct"]))

    runtime_notes.append(
        "No-provider mode tightened the card to EV >= "
        f"{effective['min_ev_pct']:.2f}%, edge >= {effective['min_edge_pct']:.2f}%, "
        f"true probability >= {effective['min_true_probability_pct']:.2f}%, "
        f"quality >= {effective['quality_floor']}, "
        f"and daily risk cap <= {effective['max_daily_risk_pct'] * 100.0:.2f}%."
    )
    if int(config.get("max_positive_odds_without_provider", 0) or 0) > 0:
        runtime_notes.append(
            f"No-provider mode blocks official picks above +{int(config.get('max_positive_odds_without_provider', 0))} unless live provider support is available."
        )
    effective["_runtime_notes"] = runtime_notes
    return effective


def rebalance_shortlist_stakes(shortlist: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    if not shortlist:
        return shortlist

    bankroll_base = max(item.get("bankrollBase", 0.0) for item in shortlist)
    if bankroll_base <= 0.0:
        return shortlist

    max_total_stake = bankroll_base * float(config.get("max_daily_risk_pct", 0.12))
    current_total_stake = sum(item.get("stake", 0.0) for item in shortlist)
    if current_total_stake <= 0.0 or current_total_stake <= max_total_stake:
        for item in shortlist:
            item["daily_risk_scaled"] = False
        return shortlist

    scale = max_total_stake / current_total_stake
    for item in shortlist:
        item["stake"] = round(item["stake"] * scale, 2)
        item["stake_pct"] = round((item["stake"] / bankroll_base) * 100.0, 2) if bankroll_base else 0.0
        item["daily_risk_scaled"] = True
    return shortlist

def build_readable_card(
    shortlist: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    ledger_state: dict[str, float],
    model_performance: dict[str, Any],
    provider_report: dict[str, Any],
    config: dict[str, Any],
    day_iso: str,
) -> str:
    planned_stake = sum(item.get("stake", 0.0) for item in shortlist)
    daily_risk_limit = ledger_state["current_bankroll"] * float(config.get("max_daily_risk_pct", 0.12))
    scaled_count = sum(1 for item in shortlist if item.get("daily_risk_scaled"))
    provider_status = "synced" if provider_report.get("synced") else "not synced"
    provider_name = provider_report.get("provider") or "none"
    provider_warning = "; ".join(provider_report.get("warnings", [])[:2])
    provider_quota = provider_report.get("requests_remaining", "")
    header = [
        f"# Daily Top Bets - {day_iso}",
        "",
        f"Starting bankroll: ${config['bankroll']:.2f}",
        f"Current bankroll from settled ledger ({ledger_state.get('ledger_source', 'bet_results')}): ${ledger_state['current_bankroll']:.2f}",
        f"Current drawdown: {ledger_state['current_drawdown_pct']:.2f}%",
        f"Tracked model ROI: {model_performance['roi_pct']:.2f}% across {model_performance['settled_bets']} settled official picks",
        f"Tracked model CLV: {model_performance['avg_clv_pct']:.2f}% average with {model_performance['open_bets']} open picks",
        f"Planned stake today: ${planned_stake:.2f} against daily risk cap ${daily_risk_limit:.2f}",
        f"Daily exposure scaling applied to {scaled_count} picks",
        f"Provider feed: {provider_status} via {provider_name} with {provider_report.get('markets_written', 0)} markets from {provider_report.get('events_seen', 0)} events",
        "Stake sizing uses the current bankroll and applies history guardrails when a sport or market has weak settled CLV/ROI.",
        "",
    ]
    if provider_quota:
        header.extend([f"Provider requests remaining: {provider_quota}", ""])
    if provider_warning:
        header.extend([f"Provider notes: {provider_warning}", ""])
    if config.get("_provider_safe_mode"):
        header.extend(["Provider-safe mode: live provider confirmation is unavailable, so the card relies on the public-board model and existing risk filters without forcing a pick count."])
        for note in config.get("_runtime_notes", []):
            header.append(note)
        header.append("")
    watchlist = build_watchlist(candidates, config)
    if not shortlist:
        lines = header + ["No official bets qualified under the current thresholds."]
        if watchlist:
            lines.extend(["", "Closest near misses (not official bets):", ""])
            for idx, item in enumerate(watchlist, start=1):
                lines.extend([
                    f"{idx}. {item['sport']} - {item['event']}",
                    f"Bet type: {item['betType']}",
                    f"Odds: {format_american(item['odds'])}",
                    f"TRUE probability: {item['true_probability_pct']:.2f}%",
                    f"EV: {item['ev_pct']:.2f}%",
                    f"Edge: {item['edge_pct']:.2f}%",
                    f"Reasoning: {item['notes']}",
                    "",
                ])
        return "\n".join(lines).rstrip() + "\n"

    lines = header
    for idx, item in enumerate(shortlist, start=1):
        lines.extend([
            f"{idx}. {item['sport']} - {item['event']}",
            f"Bet type: {item['betType']}",
            f"Odds: {format_american(item['odds'])}",
            f"Implied probability: {item['implied_probability_pct']:.2f}%",
            f"TRUE probability: {item['true_probability_pct']:.2f}%",
            f"EV: {item['ev_pct']:.2f}%",
            f"Edge score: {int(item['edge_score'])}",
            f"Stake: ${item['stake']:.2f}",
            f"Maximum acceptable odds: {format_american(item['max_odds'])}",
            f"Reasoning: {item['notes']}",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def build_phone_text_card(
    shortlist: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    ledger_state: dict[str, float],
    model_performance: dict[str, Any],
    provider_report: dict[str, Any],
    config: dict[str, Any],
    day_iso: str,
) -> str:
    watchlist = build_watchlist(candidates, config)
    lines = [
        f"EDGE LEDGER PHONE CARD - {day_iso}",
        "",
        f"Official picks: {len(shortlist)}",
        f"Bankroll ({ledger_state.get('ledger_source', 'bet_results')}): ${ledger_state['current_bankroll']:.2f}",
        f"Model ROI: {model_performance['roi_pct']:.2f}%",
        f"Settled picks: {model_performance['settled_bets']}",
        f"Open picks: {model_performance['open_bets']}",
        f"Provider synced: {'Yes' if provider_report.get('synced') else 'No'}",
    ]
    provider_notes = "; ".join(provider_report.get("warnings", [])[:2])
    if provider_notes:
        lines.extend(["", f"Provider notes: {provider_notes}"])
    if config.get("_provider_safe_mode"):
        lines.extend(["", "Provider-safe mode: public-board model only, no hard pick cap."])
        for note in config.get("_runtime_notes", []):
            lines.append(note)

    if shortlist:
        lines.extend(["", "TODAY'S OFFICIAL CARD", ""])
        for index, item in enumerate(shortlist, start=1):
            lines.extend(
                [
                    f"{index}. {item['sport']} - {item['event']}",
                    f"   {item['betType']} @ {format_american(item['odds'])}",
                    f"   True {item['true_probability_pct']:.2f}% | EV {item['ev_pct']:.2f}% | Stake ${item['stake']:.2f}",
                    f"   Max odds {format_american(item['max_odds'])}",
                    f"   {item['notes']}",
                    "",
                ]
            )
    else:
        lines.extend(["", "No official bets qualified today.", ""])

    if watchlist:
        lines.extend(["WATCHLIST", ""])
        for index, item in enumerate(watchlist, start=1):
            lines.extend(
                [
                    f"{index}. {item['sport']} - {item['betType']} @ {format_american(item['odds'])}",
                    f"   True {item['true_probability_pct']:.2f}% | EV {item['ev_pct']:.2f}% | Edge {item['edge_pct']:.2f}%",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def build_phone_html_report(
    shortlist: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    ledger_state: dict[str, float],
    model_performance: dict[str, Any],
    provider_report: dict[str, Any],
    config: dict[str, Any],
    day_iso: str,
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %I:%M %p")
    watchlist = build_watchlist(candidates, config)
    provider_notes = html_escape("; ".join(provider_report.get("warnings", [])[:2]) or "No provider warnings.")
    provider_status = "Live-confirmed" if provider_report.get("synced") else "Model-only safe mode"
    provider_safe_note = ""
    if config.get("_provider_safe_mode"):
        runtime_summary = " ".join(config.get("_runtime_notes", []))
        provider_safe_note = "Live provider confirmation is unavailable, so this phone snapshot is using the public-board model without a hard pick cap."
        if runtime_summary:
            provider_safe_note = f"{provider_safe_note} {runtime_summary}"
    pick_cards = "\n".join(render_phone_pick_card(index + 1, item) for index, item in enumerate(shortlist))
    if not pick_cards:
        pick_cards = """
        <article class="phone-empty">
          <h2>No official bets today</h2>
          <p>The filters did not find a strong enough edge.</p>
        </article>
        """
    watch_cards = "\n".join(render_phone_watch_card(index + 1, item) for index, item in enumerate(watchlist))
    if not watch_cards:
        watch_cards = '<article class="phone-watch-empty"><p>No near misses cleared the watchlist floor.</p></article>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Phone Daily Card - {day_iso}</title>
  <style>
    :root {{
      --bg: #f4ecdd;
      --paper: #fffaf1;
      --ink: #192218;
      --muted: #667265;
      --green: #25523a;
      --gold: #b6822f;
      --line: rgba(25, 34, 24, 0.12);
      --shadow: 0 14px 28px rgba(60, 44, 15, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: 'Segoe UI', Arial, sans-serif;
      background: linear-gradient(180deg, #f8f2e7, var(--bg));
      color: var(--ink);
    }}
    main {{
      width: min(100%, 560px);
      margin: 0 auto;
      padding: 14px 12px 30px;
    }}
    .hero, .panel, .pick, .watch, .phone-empty, .phone-watch-empty {{
      background: rgba(255, 250, 241, 0.96);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: var(--shadow);
    }}
    .hero {{
      padding: 18px 16px;
      margin-bottom: 14px;
    }}
    .eyebrow {{
      margin: 0 0 8px;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      font-size: 11px;
      font-weight: 700;
      color: var(--green);
    }}
    h1 {{
      margin: 0;
      font-size: 38px;
      line-height: 0.92;
    }}
    .hero p {{
      margin: 10px 0 0;
      line-height: 1.55;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }}
    .mini {{
      padding: 10px 12px;
      border-radius: 16px;
      background: rgba(37, 82, 58, 0.06);
    }}
    .mini span {{
      display: block;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }}
    .mini strong {{
      display: block;
      margin-top: 4px;
      font-size: 18px;
    }}
    .section-title {{
      margin: 18px 4px 10px;
      font-size: 17px;
      font-weight: 700;
    }}
    .pick, .watch {{
      padding: 16px 14px;
      margin-bottom: 12px;
    }}
    .rank {{
      display: inline-block;
      min-width: 32px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(182,130,47,0.18);
      color: var(--green);
      font-weight: 700;
      font-size: 12px;
      text-align: center;
    }}
    .sport {{
      margin: 12px 0 4px;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--green);
      font-weight: 700;
    }}
    .event {{
      margin: 0;
      font-size: 23px;
      line-height: 1.05;
    }}
    .bet {{
      margin: 10px 0 12px;
      font-size: 18px;
      font-weight: 700;
    }}
    .reason {{
      margin: 12px 0 0;
      color: var(--muted);
      line-height: 1.55;
      font-size: 14px;
    }}
    .panel {{
      padding: 14px 16px;
      margin-top: 16px;
    }}
    .status {{
      color: var(--green);
      font-weight: 700;
    }}
    .foot {{
      margin-top: 16px;
      font-size: 12px;
      color: var(--muted);
      line-height: 1.6;
      text-align: center;
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <p class="eyebrow">Phone Daily Card</p>
      <h1>Edge Ledger</h1>
      <p>Open this file from OneDrive on your phone to see the latest official card without using the desktop app.</p>
      <div class="grid">
        <div class="mini"><span>Date</span><strong>{html_escape(day_iso)}</strong></div>
        <div class="mini"><span>Official Picks</span><strong>{len(shortlist)}</strong></div>
        <div class="mini"><span>Bankroll</span><strong>${ledger_state['current_bankroll']:.2f}</strong></div>
        <div class="mini"><span>Ledger</span><strong>{html_escape(str(ledger_state.get('ledger_source', 'bet_results')))}</strong></div>
        <div class="mini"><span>Provider</span><strong>{html_escape(provider_status)}</strong></div>
        <div class="mini"><span>Open Picks</span><strong>{model_performance['open_bets']}</strong></div>
      </div>
    </section>

    <div class="section-title">Today's Official Card</div>
    {pick_cards}

    <section class="panel">
      <div class="status">Provider notes</div>
      <p>{provider_notes}</p>
      <p>{html_escape(provider_safe_note) if provider_safe_note else ''}</p>
    </section>

    <div class="section-title">Watchlist</div>
    {watch_cards}

    <p class="foot">Generated {html_escape(generated_at)}. This phone card is written automatically every time the daily model runs.</p>
  </main>
</body>
</html>
"""


def build_html_report(
    shortlist: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    ledger_state: dict[str, float],
    model_performance: dict[str, Any],
    provider_report: dict[str, Any],
    config: dict[str, Any],
    day_iso: str,
) -> str:
    qualified_count = sum(1 for item in candidates if item["qualified"])
    watchlist = build_watchlist(candidates, config)
    generated_at = datetime.now().strftime("%Y-%m-%d %I:%M %p")
    planned_stake = sum(item.get("stake", 0.0) for item in shortlist)
    daily_risk_limit = ledger_state["current_bankroll"] * float(config.get("max_daily_risk_pct", 0.12))
    scaled_count = sum(1 for item in shortlist if item.get("daily_risk_scaled"))
    filter_summary = summarize_filter_reasons(candidates)
    provider_status = "Synced" if provider_report.get("synced") else "Not synced"
    provider_notes = html_escape("; ".join(provider_report.get("warnings", [])[:2]) or "No provider warnings.")
    provider_quota = html_escape(str(provider_report.get("requests_remaining", "")))
    provider_safe_note = ""
    if config.get("_provider_safe_mode"):
        provider_safe_note = " Provider-safe mode is active, so this card is using the public-board model without a hard pick cap."
        runtime_summary = " ".join(config.get("_runtime_notes", []))
        if runtime_summary:
            provider_safe_note = f"{provider_safe_note} {runtime_summary}"
    top_cards = "\n".join(render_pick_card(index + 1, item) for index, item in enumerate(shortlist))
    if not top_cards:
        top_cards = """
        <article class="empty-card">
          <h2>No bets qualified</h2>
          <p>The model did not find enough statistically strong edges today. That is a feature, not a bug.</p>
        </article>
        """

    next_up_rows = "\n".join(render_next_up_row(item) for item in candidates[:12])
    if not next_up_rows:
        next_up_rows = '<tr><td colspan="7" class="muted-cell">No candidates available.</td></tr>'
    watchlist_rows = "\n".join(render_next_up_row(item) for item in watchlist)
    if not watchlist_rows:
        watchlist_rows = '<tr><td colspan="7" class="muted-cell">No near misses cleared the watchlist floors.</td></tr>'
    sport_rows = "\n".join(render_performance_row(item) for item in model_performance["by_sport"][:6])
    if not sport_rows:
        sport_rows = '<tr><td colspan="5" class="muted-cell">No settled model results yet.</td></tr>'
    market_rows = "\n".join(render_performance_row(item) for item in model_performance["by_market"][:6])
    if not market_rows:
        market_rows = '<tr><td colspan="5" class="muted-cell">No settled model results yet.</td></tr>'
    filter_rows = "\n".join(render_filter_row(label, count) for label, count in filter_summary[:6])
    if not filter_rows:
        filter_rows = '<tr><td colspan="2" class="muted-cell">No filters were needed.</td></tr>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Daily Edge Card - {day_iso}</title>
  <style>
    :root {{
      --bg: #f4ecdd;
      --paper: #fffaf1;
      --ink: #192218;
      --muted: #687463;
      --line: rgba(25, 34, 24, 0.1);
      --green: #25523a;
      --gold: #b6822f;
      --red: #a34c39;
      --shadow: 0 18px 42px rgba(60, 44, 15, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: 'Segoe UI', Arial, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(182,130,47,0.16), transparent 28%),
        radial-gradient(circle at bottom right, rgba(37,82,58,0.12), transparent 30%),
        linear-gradient(180deg, #f8f2e7, var(--bg));
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 18px 48px;
    }}
    .hero {{
      background: rgba(255,250,241,0.92);
      border: 1px solid var(--line);
      border-radius: 26px;
      box-shadow: var(--shadow);
      padding: 24px;
      margin-bottom: 18px;
    }}
    .eyebrow {{
      margin: 0 0 6px;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      font-size: 12px;
      font-weight: 700;
      color: var(--green);
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: clamp(38px, 6vw, 64px);
      line-height: 0.95;
    }}
    .hero p {{
      margin: 0;
      line-height: 1.6;
      max-width: 70ch;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .hero-note {{
      margin-top: 14px !important;
      color: var(--muted);
      max-width: 100%;
    }}
    .metric {{
      background: rgba(255,255,255,0.72);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px 16px;
    }}
    .metric-label {{
      display: block;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 4px;
    }}
    .metric-value {{
      font-size: 24px;
      font-weight: 700;
    }}
    .section {{
      margin-top: 22px;
    }}
    .section h2 {{
      margin: 0 0 10px;
      font-size: 26px;
    }}
    .section p {{
      margin: 0 0 14px;
      color: var(--muted);
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 14px;
    }}
    .pick-card, .empty-card {{
      background: rgba(255,250,241,0.92);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: var(--shadow);
      padding: 18px;
    }}
    .pick-rank {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 38px;
      height: 38px;
      border-radius: 999px;
      background: rgba(182,130,47,0.12);
      color: var(--gold);
      font-weight: 700;
      margin-bottom: 12px;
    }}
    .sport {{
      margin: 0 0 6px;
      color: var(--green);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.14em;
    }}
    .event {{
      margin: 0 0 10px;
      font-size: 22px;
      line-height: 1.15;
    }}
    .bet {{
      margin: 0 0 12px;
      font-size: 16px;
      font-weight: 700;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }}
    .mini {{
      background: rgba(255,255,255,0.7);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 10px 12px;
    }}
    .mini span {{
      display: block;
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 4px;
      font-weight: 700;
    }}
    .mini strong {{
      font-size: 16px;
    }}
    .reason {{
      margin: 0;
      line-height: 1.55;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: rgba(255,250,241,0.92);
      border: 1px solid var(--line);
      border-radius: 18px;
      overflow: hidden;
      box-shadow: var(--shadow);
    }}
    th, td {{
      padding: 12px 14px;
      text-align: left;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    th {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      background: rgba(37,82,58,0.05);
    }}
    tr:last-child td {{ border-bottom: none; }}
    .positive {{ color: var(--green); font-weight: 700; }}
    .muted-cell {{ color: var(--muted); text-align: center; }}
    .footer {{
      margin-top: 20px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.6;
    }}
    @media (max-width: 860px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
  <body>
  <main>
    <section class="hero">
      <p class="eyebrow">One-Click Daily Betting Card</p>
      <h1>Daily Edge Card</h1>
      <p>The model scanned the configured ESPN league boards, synced the live provider feed when credentials were available, and then kept only the bets that still cleared the EV, edge, and quality filters after calibration, uncertainty cuts, bankroll discipline, and historical CLV/ROI guardrails.</p>
      <div class="summary">
        <div class="metric"><span class="metric-label">Date</span><div class="metric-value">{html_escape(day_iso)}</div></div>
        <div class="metric"><span class="metric-label">Qualified Bets</span><div class="metric-value">{qualified_count}</div></div>
        <div class="metric"><span class="metric-label">Final Picks</span><div class="metric-value">{len(shortlist)}</div></div>
        <div class="metric"><span class="metric-label">Current Bankroll</span><div class="metric-value">${ledger_state['current_bankroll']:.2f}</div></div>
        <div class="metric"><span class="metric-label">Ledger Source</span><div class="metric-value">{html_escape(str(ledger_state.get('ledger_source', 'bet_results')))}</div></div>
        <div class="metric"><span class="metric-label">Model ROI</span><div class="metric-value">{model_performance['roi_pct']:.2f}%</div></div>
        <div class="metric"><span class="metric-label">Settled Picks</span><div class="metric-value">{model_performance['settled_bets']}</div></div>
        <div class="metric"><span class="metric-label">Open Picks</span><div class="metric-value">{model_performance['open_bets']}</div></div>
        <div class="metric"><span class="metric-label">Avg CLV</span><div class="metric-value">{model_performance['avg_clv_pct']:.2f}%</div></div>
        <div class="metric"><span class="metric-label">Planned Stake</span><div class="metric-value">${planned_stake:.2f}</div></div>
        <div class="metric"><span class="metric-label">Daily Risk Cap</span><div class="metric-value">${daily_risk_limit:.2f}</div></div>
        <div class="metric"><span class="metric-label">Scaled Picks</span><div class="metric-value">{scaled_count}</div></div>
        <div class="metric"><span class="metric-label">Provider Feed</span><div class="metric-value">{provider_status}</div></div>
        <div class="metric"><span class="metric-label">Provider Markets</span><div class="metric-value">{provider_report.get('markets_written', 0)}</div></div>
        <div class="metric"><span class="metric-label">Provider Events</span><div class="metric-value">{provider_report.get('events_seen', 0)}</div></div>
        <div class="metric"><span class="metric-label">Provider</span><div class="metric-value">{html_escape(provider_report.get('provider') or 'none')}</div></div>
        <div class="metric"><span class="metric-label">Requests Left</span><div class="metric-value">{provider_quota or '&mdash;'}</div></div>
      </div>
      <p class="hero-note">Generated {generated_at}. Provider notes: {provider_notes}{html_escape(provider_safe_note)}</p>
    </section>

    <section class="section">
      <h2>Today's Card</h2>
      <p>If there are fewer than five picks, the model is choosing discipline over forcing volume.</p>
      <div class="cards">
        {top_cards}
      </div>
    </section>

    <section class="section">
      <h2>Model Track Record</h2>
      <p>The system now auto-settles official game-line picks from ESPN finals and keeps an always-on performance ledger for inspection.</p>
      <div class="cards">
        <article class="pick-card">
          <p class="sport">Overall</p>
          <h3 class="event">Official Picks</h3>
          <div class="grid">
            <div class="mini"><span>Profit</span><strong>${model_performance['profit']:.2f}</strong></div>
            <div class="mini"><span>ROI</span><strong>{model_performance['roi_pct']:.2f}%</strong></div>
            <div class="mini"><span>Hit Rate</span><strong>{model_performance['hit_rate_pct']:.2f}%</strong></div>
            <div class="mini"><span>Avg EV</span><strong>{model_performance['avg_ev_pct']:.2f}%</strong></div>
            <div class="mini"><span>Wins-Losses</span><strong>{model_performance['wins']}-{model_performance['losses']}</strong></div>
            <div class="mini"><span>Pushes</span><strong>{model_performance['pushes']}</strong></div>
          </div>
          <p class="reason">Tracked from <code>tracking/model_results.csv</code> and updated automatically when final scores are available.</p>
        </article>
      </div>
    </section>

    <section class="section">
      <h2>Performance Breakdown</h2>
      <p>Breakdowns make it easier to see whether a specific sport or market type is dragging down long-term results.</p>
      <table>
        <thead>
          <tr>
            <th>Sport</th>
            <th>Bets</th>
            <th>ROI %</th>
            <th>Profit</th>
            <th>Avg CLV %</th>
          </tr>
        </thead>
        <tbody>
          {sport_rows}
        </tbody>
      </table>
      <div style="height: 14px;"></div>
      <table>
        <thead>
          <tr>
            <th>Market</th>
            <th>Bets</th>
            <th>ROI %</th>
            <th>Profit</th>
            <th>Avg CLV %</th>
          </tr>
        </thead>
        <tbody>
          {market_rows}
        </tbody>
      </table>
    </section>

    <section class="section">
      <h2>Filter Diagnostics</h2>
      <p>When the board is empty, this shows the most common reasons candidates were rejected.</p>
      <table>
        <thead>
          <tr>
            <th>Filter Reason</th>
            <th>Count</th>
          </tr>
        </thead>
        <tbody>
          {filter_rows}
        </tbody>
      </table>
    </section>

    <section class="section">
      <h2>Closest Near Misses</h2>
      <p>These are not official bets. They were the strongest filtered candidates that still carried some positive edge after the new safety checks.</p>
      <table>
        <thead>
          <tr>
            <th>Bet</th>
            <th>Odds</th>
            <th>True %</th>
            <th>EV %</th>
            <th>Edge %</th>
            <th>Score</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {watchlist_rows}
        </tbody>
      </table>
    </section>

    <section class="section">
      <h2>Next Best Candidates</h2>
      <p>This is the top of the full ranked board, including bets that may have failed the final qualification rules.</p>
      <table>
        <thead>
          <tr>
            <th>Bet</th>
            <th>Odds</th>
            <th>True %</th>
            <th>EV %</th>
            <th>Edge %</th>
            <th>Score</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {next_up_rows}
        </tbody>
      </table>
    </section>

    <p class="footer">
      Generated at {html_escape(generated_at)}. Settings come from edge_model_config.json. Settled bankroll state comes from tracking/bet_results.csv. Official model performance comes from tracking/model_results.csv.
    </p>
  </main>
</body>
</html>
"""


def automation_shape(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "date": item["date"],
        "sport": item["sport"],
        "event": item["event"],
        "bet_type": item["betType"],
        "odds": format_american(item["odds"]),
        "implied_probability": f"{item['implied_probability_pct']:.2f}%",
        "true_probability": f"{item['true_probability_pct']:.2f}%",
        "ev": f"{item['ev_pct']:.2f}%",
        "edge_score": str(int(item["edge_score"])),
        "stake": f"${item['stake']:.2f}",
        "max_odds": format_american(item["max_odds"]),
    }


def render_pick_card(rank: int, item: dict[str, Any]) -> str:
    return f"""
    <article class="pick-card">
      <div class="pick-rank">{rank}</div>
      <p class="sport">{html_escape(item['sport'])}</p>
      <h3 class="event">{html_escape(item['event'])}</h3>
      <p class="bet">{html_escape(item['betType'])} @ {html_escape(format_american(item['odds']))}</p>
      <div class="grid">
        <div class="mini"><span>Implied</span><strong>{item['implied_probability_pct']:.2f}%</strong></div>
        <div class="mini"><span>True</span><strong>{item['true_probability_pct']:.2f}%</strong></div>
        <div class="mini"><span>EV</span><strong>{item['ev_pct']:.2f}%</strong></div>
        <div class="mini"><span>Stake</span><strong>${item['stake']:.2f}</strong></div>
        <div class="mini"><span>Edge Score</span><strong>{int(item['edge_score'])}</strong></div>
        <div class="mini"><span>Max Odds</span><strong>{html_escape(format_american(item['max_odds']))}</strong></div>
      </div>
      <p class="reason">{html_escape(item['notes'])}</p>
    </article>
    """


def render_phone_pick_card(rank: int, item: dict[str, Any]) -> str:
    return f"""
    <article class="pick">
      <div class="rank">#{rank}</div>
      <p class="sport">{html_escape(item['sport'])}</p>
      <h2 class="event">{html_escape(item['event'])}</h2>
      <p class="bet">{html_escape(item['betType'])} @ {html_escape(format_american(item['odds']))}</p>
      <div class="grid">
        <div class="mini"><span>Implied</span><strong>{item['implied_probability_pct']:.2f}%</strong></div>
        <div class="mini"><span>True</span><strong>{item['true_probability_pct']:.2f}%</strong></div>
        <div class="mini"><span>EV</span><strong>{item['ev_pct']:.2f}%</strong></div>
        <div class="mini"><span>Stake</span><strong>${item['stake']:.2f}</strong></div>
        <div class="mini"><span>Edge</span><strong>{item['edge_pct']:.2f}%</strong></div>
        <div class="mini"><span>Max Odds</span><strong>{html_escape(format_american(item['max_odds']))}</strong></div>
      </div>
      <p class="reason">{html_escape(item['notes'])}</p>
    </article>
    """


def render_phone_watch_card(rank: int, item: dict[str, Any]) -> str:
    return f"""
    <article class="watch">
      <div class="rank">W{rank}</div>
      <p class="sport">{html_escape(item['sport'])}</p>
      <h2 class="event">{html_escape(item['betType'])}</h2>
      <p class="bet">{html_escape(item['event'])}</p>
      <div class="grid">
        <div class="mini"><span>Odds</span><strong>{html_escape(format_american(item['odds']))}</strong></div>
        <div class="mini"><span>True</span><strong>{item['true_probability_pct']:.2f}%</strong></div>
        <div class="mini"><span>EV</span><strong>{item['ev_pct']:.2f}%</strong></div>
        <div class="mini"><span>Edge</span><strong>{item['edge_pct']:.2f}%</strong></div>
      </div>
    </article>
    """


def render_next_up_row(item: dict[str, Any]) -> str:
    status = "Qualified" if item["qualified"] else "Filtered"
    status_class = "positive" if item["qualified"] else ""
    return f"""
    <tr>
      <td><strong>{html_escape(item['betType'])}</strong><br><span>{html_escape(item['sport'])} - {html_escape(item['event'])}</span></td>
      <td>{html_escape(format_american(item['odds']))}</td>
      <td>{item['true_probability_pct']:.2f}%</td>
      <td>{item['ev_pct']:.2f}%</td>
      <td>{item['edge_pct']:.2f}%</td>
      <td>{int(round(item['composite_score']))}</td>
      <td class="{status_class}">{status}</td>
    </tr>
    """


def render_performance_row(item: dict[str, Any]) -> str:
    return f"""
    <tr>
      <td><strong>{html_escape(item['label'])}</strong></td>
      <td>{item['bets']}</td>
      <td>{item['roi_pct']:.2f}%</td>
      <td>${item['profit']:.2f}</td>
      <td>{item['avg_clv_pct']:.2f}%</td>
    </tr>
    """


def render_filter_row(label: str, count: int) -> str:
    return f"""
    <tr>
      <td><strong>{html_escape(label)}</strong></td>
      <td>{count}</td>
    </tr>
    """


def build_watchlist(candidates: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    limit = int(config.get("watchlist_size", 5))
    if limit <= 0:
        return []

    min_ev = float(config.get("watchlist_min_ev_pct", 2.0))
    min_edge = float(config.get("watchlist_min_edge_pct", 1.0))
    min_true = float(config.get("watchlist_min_true_probability_pct", 20.0))
    watchlist = [
        item
        for item in candidates
        if not item.get("qualified")
        and item.get("ev_pct", 0.0) >= min_ev
        and item.get("edge_pct", 0.0) >= min_edge
        and item.get("true_probability_pct", 0.0) >= min_true
    ]
    return watchlist[:limit]


def append_recommendations_log(shortlist: list[dict[str, Any]], day_iso: str) -> None:
    fieldnames = [
        "date",
        "bet_id",
        "sport",
        "event",
        "event_id",
        "market_type",
        "bet_type",
        "selection",
        "line",
        "odds",
        "true_probability",
        "ev_pct",
        "stake",
        "stake_pct",
        "edge_score",
        "history_penalty_pct",
        "notes",
    ]
    existing_rows = [
        normalize_tracking_row(row)
        for row in read_csv_rows(RECOMMENDATIONS_LOG_PATH)
        if row.get("date", "") != day_iso
    ]
    existing_rows.extend(build_recommendation_log_row(item) for item in shortlist)
    existing_rows.sort(key=lambda row: (row.get("date", ""), row.get("bet_id", "")))

    with RECOMMENDATIONS_LOG_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)


def sync_model_results_snapshot(shortlist: list[dict[str, Any]], day_iso: str) -> None:
    fieldnames = [
        "bet_id",
        "date",
        "placed_date",
        "settled_date",
        "sport",
        "event",
        "event_id",
        "market_type",
        "source",
        "bet_type",
        "selection",
        "selection_team",
        "line",
        "line_value",
        "home_team",
        "away_team",
        "odds",
        "closing_odds",
        "closing_line",
        "stake",
        "true_probability",
        "ev_pct",
        "edge_score",
        "history_penalty_pct",
        "sportsbook",
        "result",
        "profit",
        "clv_pct",
        "home_score",
        "away_score",
        "notes",
    ]
    existing_rows = [
        normalize_tracking_row(row)
        for row in read_csv_rows(MODEL_RESULTS_PATH)
        if row.get("date", "") != day_iso or is_settled_result(row)
    ]
    existing_rows.extend(build_model_result_row(item, day_iso) for item in shortlist)
    existing_rows.sort(key=lambda row: (row.get("date", ""), row.get("bet_id", "")))

    with MODEL_RESULTS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)


def build_recommendation_log_row(item: dict[str, Any]) -> dict[str, str]:
    snapshot = infer_snapshot_fields(item)
    return {
        "date": item["date"],
        "bet_id": item["id"],
        "sport": item["sport"],
        "event": item["event"],
        "event_id": str(item.get("eventId", "")).strip() or item["correlationGroup"],
        "market_type": item["marketType"],
        "bet_type": item["betType"],
        "selection": snapshot["selection"],
        "line": item["line"],
        "odds": format_american(item["odds"]),
        "true_probability": f"{item['true_probability_pct']:.2f}%",
        "ev_pct": f"{item['ev_pct']:.2f}%",
        "stake": f"{item['stake']:.2f}",
        "stake_pct": f"{item['stake_pct']:.2f}%",
        "edge_score": str(int(item["edge_score"])),
        "history_penalty_pct": f"{item['history_penalty_pct']:.2f}%",
        "notes": item["notes"],
    }


def build_model_result_row(item: dict[str, Any], day_iso: str) -> dict[str, str]:
    snapshot = infer_snapshot_fields(item)
    return {
        "bet_id": item["id"],
        "date": day_iso,
        "placed_date": day_iso,
        "settled_date": "",
        "sport": item["sport"],
        "event": item["event"],
        "event_id": str(item.get("eventId", "")).strip() or item["correlationGroup"],
        "market_type": item["marketType"],
        "source": str(item.get("source", "espn")).strip() or "espn",
        "bet_type": item["betType"],
        "selection": snapshot["selection"],
        "selection_team": snapshot["selection_team"],
        "line": item["line"],
        "line_value": snapshot["line_value"],
        "home_team": snapshot["home_team"],
        "away_team": snapshot["away_team"],
        "odds": format_american(item["odds"]),
        "closing_odds": "",
        "closing_line": "",
        "stake": f"{item['stake']:.2f}",
        "true_probability": f"{item['true_probability_pct']:.2f}%",
        "ev_pct": f"{item['ev_pct']:.2f}%",
        "edge_score": str(int(item["edge_score"])),
        "history_penalty_pct": f"{item['history_penalty_pct']:.2f}%",
        "sportsbook": item["sportsbook"],
        "result": "open",
        "profit": "",
        "clv_pct": "",
        "home_score": "",
        "away_score": "",
        "notes": item["notes"],
    }


def infer_snapshot_fields(item: dict[str, Any]) -> dict[str, str]:
    away_team = str(item.get("awayTeam", "")).strip()
    home_team = str(item.get("homeTeam", "")).strip()
    if not away_team and not home_team:
        away_team, home_team = parse_event_teams(item["event"])
    market_type = item["marketType"]
    bet_type = item["betType"]
    selection = str(item.get("selection", "")).strip().lower()
    selection_team = str(item.get("selectionTeam", "")).strip()
    line_value = ""

    if market_type == "moneyline":
        if not selection_team:
            selection_team = bet_type.removesuffix(" moneyline").strip()
        if not selection:
            if selection_team == home_team:
                selection = "home"
            elif selection_team == away_team:
                selection = "away"
    elif market_type == "spread":
        line_value = format_float(parse_numeric(item.get("line", "")))
        if not selection:
            if home_team and bet_type.startswith(home_team):
                selection = "home"
                selection_team = home_team
            elif away_team and bet_type.startswith(away_team):
                selection = "away"
                selection_team = away_team
    elif market_type == "total":
        line_value = format_float(parse_numeric(item.get("line", "")))
        if not selection:
            if bet_type.lower().startswith("over"):
                selection = "over"
            elif bet_type.lower().startswith("under"):
                selection = "under"
    elif market_type in {"player_prop", "team_prop", "other"}:
        line_value = format_float(parse_numeric(item.get("line", "")))

    return {
        "home_team": home_team,
        "away_team": away_team,
        "selection": selection,
        "selection_team": selection_team,
        "line_value": line_value,
    }


def parse_event_teams(event_name: str) -> tuple[str, str]:
    for separator in (" at ", " vs. ", " vs "):
        if separator in event_name:
            away, home = event_name.split(separator, 1)
            return away.strip(), home.strip()
    return "", ""


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def is_settled_result(row: dict[str, str]) -> bool:
    result = str(row.get("result", "")).strip().lower()
    return bool(row.get("settled_date")) or result in {"win", "loss", "push"}


def extract_team_context(competition: dict[str, Any], side: str, sport: str) -> dict[str, Any] | None:
    competitors = competition.get("competitors", [])
    team = next((item for item in competitors if item.get("homeAway") == side), None)
    if not team:
        return None
    if sport in SOCCER_SPORTS:
        records = {record.get("name", "").lower(): parse_soccer_record(record.get("summary", "")) for record in team.get("records", [])}
    else:
        records = {record.get("name", "").lower(): parse_record(record.get("summary", "")) for record in team.get("records", [])}
    stats = {item.get("name"): parse_numeric(item.get("displayValue")) for item in team.get("statistics", []) if item.get("name")}
    probable_record = safe_get(team, "probables", 0, "record", default="")
    probable_name = safe_get(team, "probables", 0, "athlete", "displayName", default="")

    overall = records.get("overall") or records.get("total") or records.get("all splits") or {}
    home_record = records.get("home") or {}
    road_record = records.get("road") or {}
    last_10 = records.get("last 10") or records.get("last 10 games") or {}
    last_5 = records.get("last 5") or records.get("last 5 games") or {}

    context = {
        "team": team.get("team", {}).get("displayName", ""),
        "games": max(int(overall.get("games", 0)), 1),
        "win_pct": overall.get("win_pct", 0.5),
        "home_pct": home_record.get("win_pct", overall.get("win_pct", 0.5)),
        "road_pct": road_record.get("win_pct", overall.get("win_pct", 0.5)),
        "points_pct": overall.get("points_pct", overall.get("win_pct", 0.5)),
        "last_10_win_pct": last_10.get("win_pct", overall.get("win_pct", 0.5)),
        "last_5_win_pct": last_5.get("win_pct", overall.get("win_pct", 0.5)),
        "starter_name": probable_name,
        "starter_era": parse_probable_metric(probable_record, fallback=4.1),
        "runs_pg": 4.4,
        "team_era": 4.1,
        "avg_points": 112.0,
        "fg_pct": 46.0,
        "three_pct": 35.0,
        "avg_assists": 25.0,
        "avg_rebounds": 44.0,
        "fga": 88.0,
        "goals_pg": 3.0,
        "save_pct": 0.900,
        "shots_on_target_pg": 4.8,
        "possession_pct": 50.0,
        # Schedule analysis factors (default values)
        "rest_days": 1.0,  # Default 1 day of rest
        "travel_distance": 0.0,  # Default no travel
        "back_to_back": False,  # Default not back-to-back
        "road_trip_length": 0,  # Default not on road trip
        # Injury/roster analysis factors (default values)
        "key_injuries": 0,  # Number of key players injured
        "injury_impact": 0.0,  # Impact of injuries (0-1 scale)
        "roster_stability": 1.0,  # Roster stability (0-1 scale)
        # Head-to-head historical data (default values)
        "h2h_home_win_pct": 0.5,  # Home team win % in head-to-head
        "h2h_recent_win_pct": 0.5,  # Home team win % in last 5 head-to-head
        "h2h_total_avg": 0.0,  # Average total in head-to-head games
        # Weather factors for outdoor sports (default values)
        "temperature": 70.0,  # Default temperature in Fahrenheit
        "wind_speed": 0.0,  # Wind speed in mph
        "wind_direction": "",  # Wind direction
        "precipitation": 0.0,  # Precipitation probability (0-1)
        "humidity": 50.0,  # Humidity percentage
    }

    if sport in BASEBALL_SPORTS:
        context["runs_pg"] = stats.get("runs", 39.6) / context["games"]
        context["team_era"] = stats.get("ERA", 4.1)
    elif sport in BASKETBALL_SPORTS:
        context["avg_points"] = stats.get("avgPoints", stats.get("points", 112.0))
        context["fg_pct"] = stats.get("fieldGoalPct", 46.0)
        context["three_pct"] = stats.get("threePointPct", stats.get("threePointFieldGoalPct", 35.0))
        context["avg_assists"] = stats.get("avgAssists", 25.0)
        context["avg_rebounds"] = stats.get("avgRebounds", 44.0)
        context["fga"] = stats.get("fieldGoalsAttempted", 88.0) / context["games"]
    elif sport in HOCKEY_SPORTS:
        context["goals_pg"] = stats.get("goals", 246.0) / context["games"]
        context["save_pct"] = stats.get("savePct", 0.900)
    elif sport in SOCCER_SPORTS:
        # Use reasonable defaults when ESPN doesn't provide soccer stats
        total_goals = stats.get("totalGoals", 0.0)
        if total_goals > 0 and context["games"] > 0:
            context["goals_pg"] = total_goals / context["games"]
        else:
            context["goals_pg"] = 1.4  # Reasonable default for soccer
            context["goals_pg_defaulted"] = True
        shots_on_target = stats.get("shotsOnTarget", 0.0)
        if shots_on_target > 0 and context["games"] > 0:
            context["shots_on_target_pg"] = shots_on_target / context["games"]
        else:
            context["shots_on_target_pg"] = 4.7  # Reasonable default for soccer
            context["shots_on_target_pg_defaulted"] = True
        context["possession_pct"] = stats.get("possessionPct", 50.0)

    # Add flag to track if any stats were defaulted
    context["stats_defaulted"] = any(
        context.get(f"{key}_defaulted", False)
        for key in ["goals_pg", "shots_on_target_pg", "avg_points", "fg_pct", "goals", "save_pct"]
    )

    return context


def extract_odds(block: dict[str, Any], side: str) -> int | None:
    target = block.get(side, {})
    return extract_nested_odds(target)


def extract_total_side(block: dict[str, Any], side: str) -> dict[str, Any] | None:
    target = block.get(side, {})
    odds = extract_nested_odds(target)
    line = parse_line_value(extract_nested_line(target))
    if odds is None or line is None:
        return None
    return {"odds": odds, "line": line}


def extract_spread_side(block: dict[str, Any], side: str) -> dict[str, Any] | None:
    target = block.get(side, {})
    odds = extract_nested_odds(target)
    line = parse_line_value(extract_nested_line(target))
    if odds is None or line is None:
        return None
    return {"odds": odds, "line": line}


def extract_nested_odds(target: dict[str, Any]) -> int | None:
    for key in ("close", "open"):
        node = target.get(key, {})
        raw = node.get("odds")
        if raw is not None:
            value = parse_numeric(raw)
            if not math.isnan(value):
                return int(round(value))
    if "odds" in target and target["odds"] is not None:
        value = parse_numeric(target["odds"])
        if not math.isnan(value):
            return int(round(value))
    return None


def extract_nested_line(target: dict[str, Any]) -> str | None:
    for key in ("close", "open"):
        node = target.get(key, {})
        raw = node.get("line")
        if raw is not None:
            return str(raw)
    if "line" in target and target["line"] is not None:
        return str(target["line"])
    return None


def load_ledger_state(config: dict[str, Any]) -> dict[str, float]:
    starting_bankroll = float(config["bankroll"])
    running = starting_bankroll
    peak = starting_bankroll
    max_drawdown = 0.0
    source_name = "bet_results"

    settled_rows = load_settled_history_rows(BET_RESULTS_PATH)
    if not settled_rows:
        settled_rows = load_settled_history_rows(MODEL_RESULTS_PATH)
        if settled_rows:
            source_name = "model_results"

    for row in settled_rows:
        profit = parse_numeric(row.get("profit", ""))
        if math.isnan(profit):
            continue
        running += profit
        peak = max(peak, running)
        drawdown = ((peak - running) / peak * 100.0) if peak else 0.0
        max_drawdown = max(max_drawdown, drawdown)

    current_drawdown = ((peak - running) / peak * 100.0) if peak else 0.0
    return {
        "current_bankroll": round(running, 2),
        "current_drawdown_pct": round(current_drawdown, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "ledger_source": source_name,
    }


def load_history_profile(config: dict[str, Any]) -> dict[str, Any]:
    min_bets = int(config.get("history_min_bets", 20))
    actual_rows = load_settled_history_rows(BET_RESULTS_PATH)
    model_rows = load_settled_history_rows(MODEL_RESULTS_PATH)
    source_rows = actual_rows
    source_name = "bet_results"
    if len(actual_rows) < min_bets and len(model_rows) > len(actual_rows):
        source_rows = model_rows
        source_name = "model_results"

    profile = {
        "minimum_bets": min_bets,
        "cap_pct": float(config.get("history_penalty_cap_pct", 2.0)),
        "source": source_name,
        "sport": {},
        "market": {},
        "sport_market": {},
    }
    if not source_rows:
        return profile

    buckets: dict[tuple[str, str], dict[str, float]] = {}

    def ensure_bucket(kind: str, key: str) -> dict[str, float]:
        return buckets.setdefault(
            (kind, key),
            {
                "bets": 0.0,
                "stake_sum": 0.0,
                "profit_sum": 0.0,
                "clv_sum": 0.0,
                "clv_count": 0.0,
            },
        )

    for row in source_rows:
        sport = str(row.get("sport", "")).strip().upper()
        market_type = str(row.get("market_type", "")).strip().lower()
        if not sport or not market_type:
            continue

        stake = parse_numeric(row.get("stake", ""))
        profit = parse_numeric(row.get("profit", ""))
        clv_pct = parse_numeric(row.get("clv_pct", ""))

        for kind, key in (
            ("sport", sport),
            ("market", market_type),
            ("sport_market", f"{sport}:{market_type}"),
        ):
            bucket = ensure_bucket(kind, key)
            bucket["bets"] += 1.0
            if not math.isnan(stake):
                bucket["stake_sum"] += stake
            if not math.isnan(profit):
                bucket["profit_sum"] += profit
            if not math.isnan(clv_pct):
                bucket["clv_sum"] += clv_pct
                bucket["clv_count"] += 1.0

    for (kind, key), bucket in buckets.items():
        summary = summarize_history_bucket(bucket, config, min_bets)
        if kind == "sport":
            profile["sport"][key] = summary
        elif kind == "market":
            profile["market"][key] = summary
        else:
            profile["sport_market"][key] = summary

    return profile


def summarize_history_bucket(bucket: dict[str, float], config: dict[str, Any], min_bets: int) -> dict[str, float]:
    bets = int(bucket["bets"])
    stake_sum = bucket["stake_sum"]
    roi_pct = (bucket["profit_sum"] / stake_sum * 100.0) if stake_sum > 0 else 0.0
    avg_clv_pct = (bucket["clv_sum"] / bucket["clv_count"]) if bucket["clv_count"] > 0 else 0.0
    penalty_pct = 0.0
    boost_pct = 0.0
    if bets >= min_bets:
        if roi_pct < 0.0:
            penalty_pct += float(config.get("history_negative_roi_penalty_pct", 0.0))
        if avg_clv_pct < 0.0 and bucket["clv_count"] >= max(5.0, min_bets / 2.0):
            penalty_pct += abs(avg_clv_pct) * float(config.get("history_clv_penalty_scale", 0.0))
        if roi_pct > float(config.get("history_positive_roi_trigger_pct", 2.0)):
            boost_pct += min(
                float(config.get("history_positive_roi_boost_cap_pct", 1.5)),
                roi_pct * float(config.get("history_positive_roi_boost_scale", 0.08)),
            )
        if avg_clv_pct > float(config.get("history_positive_clv_trigger_pct", 0.5)) and bucket["clv_count"] >= max(5.0, min_bets / 2.0):
            boost_pct += min(
                float(config.get("history_positive_clv_boost_cap_pct", 1.0)),
                avg_clv_pct * float(config.get("history_positive_clv_boost_scale", 0.18)),
            )
    penalty_pct = min(penalty_pct, float(config.get("history_penalty_cap_pct", 2.0)))
    boost_pct = min(boost_pct, float(config.get("history_boost_cap_pct", 2.0)))
    return {
        "bets": bets,
        "stake_sum": round(stake_sum, 2),
        "roi_pct": round(roi_pct, 2),
        "avg_clv_pct": round(avg_clv_pct, 2),
        "penalty_pct": round(penalty_pct, 2),
        "boost_pct": round(boost_pct, 2),
    }


def history_penalty_for_market(history_profile: dict[str, Any], sport: str, market_type: str) -> dict[str, Any]:
    if not history_profile:
        return {"penalty_pct": 0.0, "boost_pct": 0.0, "summary": ""}

    penalty_candidates: list[tuple[float, str]] = []
    boost_candidates: list[tuple[float, str]] = []
    cap_pct = float(history_profile.get("cap_pct", 2.0))
    sport_market_key = f"{sport}:{market_type}"

    sport_market = history_profile.get("sport_market", {}).get(sport_market_key)
    if sport_market and sport_market["penalty_pct"] > 0.0:
        penalty_candidates.append(
            (
                sport_market["penalty_pct"],
                f"{sport} {market_type} has {sport_market['bets']} settled bets, ROI {sport_market['roi_pct']:.2f}%, CLV {sport_market['avg_clv_pct']:.2f}%",
            )
        )
    if sport_market and sport_market.get("boost_pct", 0.0) > 0.0:
        boost_candidates.append(
            (
                sport_market["boost_pct"],
                f"{sport} {market_type} is running hot with {sport_market['bets']} settled bets, ROI {sport_market['roi_pct']:.2f}%, CLV {sport_market['avg_clv_pct']:.2f}%",
            )
        )

    sport_summary = history_profile.get("sport", {}).get(sport)
    if sport_summary and sport_summary["penalty_pct"] > 0.0:
        penalty_candidates.append(
            (
                sport_summary["penalty_pct"] * 0.6,
                f"{sport} overall has {sport_summary['bets']} settled bets, ROI {sport_summary['roi_pct']:.2f}%, CLV {sport_summary['avg_clv_pct']:.2f}%",
            )
        )
    if sport_summary and sport_summary.get("boost_pct", 0.0) > 0.0:
        boost_candidates.append(
            (
                sport_summary["boost_pct"] * 0.5,
                f"{sport} overall is earning trust with {sport_summary['bets']} settled bets, ROI {sport_summary['roi_pct']:.2f}%, CLV {sport_summary['avg_clv_pct']:.2f}%",
            )
        )

    market_summary = history_profile.get("market", {}).get(market_type)
    if market_summary and market_summary["penalty_pct"] > 0.0:
        penalty_candidates.append(
            (
                market_summary["penalty_pct"] * 0.6,
                f"{market_type} overall has {market_summary['bets']} settled bets, ROI {market_summary['roi_pct']:.2f}%, CLV {market_summary['avg_clv_pct']:.2f}%",
            )
        )
    if market_summary and market_summary.get("boost_pct", 0.0) > 0.0:
        boost_candidates.append(
            (
                market_summary["boost_pct"] * 0.5,
                f"{market_type} overall is earning trust with {market_summary['bets']} settled bets, ROI {market_summary['roi_pct']:.2f}%, CLV {market_summary['avg_clv_pct']:.2f}%",
            )
        )

    penalty_pct = 0.0
    penalty_summary = ""
    if penalty_candidates:
        penalty_pct, penalty_summary = max(penalty_candidates, key=lambda item: item[0])

    boost_pct = 0.0
    boost_summary = ""
    if boost_candidates:
        boost_pct, boost_summary = max(boost_candidates, key=lambda item: item[0])

    summary_parts = []
    if penalty_summary:
        summary_parts.append(penalty_summary)
    if boost_summary:
        summary_parts.append(boost_summary)
    return {
        "penalty_pct": round(min(penalty_pct, cap_pct), 2),
        "boost_pct": round(min(boost_pct, float(history_profile.get("cap_pct", 2.0))), 2),
        "summary": " | ".join(summary_parts),
    }


def load_settled_history_rows(path: Path) -> list[dict[str, str]]:
    settled_rows: list[dict[str, str]] = []
    for row in read_csv_rows(path):
        profit = parse_numeric(row.get("profit", ""))
        if is_settled_result(row) or not math.isnan(profit):
            settled_rows.append(row)
    return settled_rows


def settle_model_results(as_of_date_iso: str) -> dict[str, int]:
    rows = [normalize_tracking_row(row) for row in read_csv_rows(MODEL_RESULTS_PATH)]
    if not rows:
        return {"settled": 0, "remaining_open": 0}

    target_date = normalize_iso_date(as_of_date_iso)
    event_cache: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    settled_count = 0
    updated_count = 0
    settlement_overrides = load_override_map(SETTLEMENT_OVERRIDES_PATH)
    closing_overrides = load_override_map(CLOSING_LINE_OVERRIDES_PATH)

    for index, row in enumerate(rows):
        override_row = settlement_overrides.get(str(row.get("bet_id", "")).strip())
        if override_row and not is_settled_result(row):
            rows[index] = apply_settlement_override(row, override_row)
            settled_count += 1
            updated_count += 1
            continue

        closing_override = closing_overrides.get(str(row.get("bet_id", "")).strip())
        if closing_override:
            updated_row = apply_closing_override(row, closing_override)
            if updated_row != row:
                rows[index] = updated_row
                updated_count += 1
                row = updated_row

        raw_date = re.sub(r"[^0-9]", "", str(row.get("date", "")))
        if len(raw_date) < 8:
            continue
        row_date = normalize_iso_date(raw_date)
        sport = str(row.get("sport", "")).strip().upper()
        if not row_date or row_date > target_date or sport not in SPORT_ENDPOINTS or is_settled_result(row):
            continue

        cache_key = (sport, row_date)
        if cache_key not in event_cache:
            try:
                payload = fetch_json(SPORT_ENDPOINTS[sport].format(date=row_date.replace("-", "")))
            except Exception:
                event_cache[cache_key] = {}
            else:
                event_cache[cache_key] = {str(event.get("id", "")): event for event in payload.get("events", [])}

        event = event_cache[cache_key].get(str(row.get("event_id", "")))
        if not event:
            event = find_matching_event(list(event_cache[cache_key].values()), row)
        if event and not is_settled_result(row):
            current_market_row = snapshot_open_market_row(row, event)
            if current_market_row != row:
                rows[index] = current_market_row
                updated_count += 1
                row = current_market_row
        settled_row = settle_model_result_row(row, event, row_date)
        if not settled_row:
            continue
        rows[index] = settled_row
        settled_count += 1
        updated_count += 1

    if updated_count:
        with MODEL_RESULTS_PATH.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    remaining_open = sum(1 for row in rows if not is_settled_result(row))
    return {"settled": settled_count, "remaining_open": remaining_open}


def settle_model_result_row(row: dict[str, str], event: dict[str, Any] | None, row_date: str) -> dict[str, str] | None:
    if not event:
        return None
    competition = safe_get(event, "competitions", 0)
    if not competition or not event_is_final(event, competition):
        return None

    home = next((item for item in competition.get("competitors", []) if item.get("homeAway") == "home"), None)
    away = next((item for item in competition.get("competitors", []) if item.get("homeAway") == "away"), None)
    if not home or not away:
        return None

    home_score = parse_numeric(home.get("score"))
    away_score = parse_numeric(away.get("score"))
    if math.isnan(home_score) or math.isnan(away_score):
        return None

    result = grade_result(row, int(home_score), int(away_score))
    if not result:
        return None

    closing_market = extract_closing_market(competition, row)
    closing_odds = closing_market.get("odds")
    closing_line = closing_market.get("line")
    stake = parse_numeric(row.get("stake", ""))
    odds = int(parse_numeric(row.get("odds", "")))
    profit = settle_profit(stake, odds, result) if not math.isnan(stake) else math.nan
    clv_pct = calculate_clv_pct(odds, closing_odds)

    settled_row = dict(row)
    settled_row.update(
        {
            "settled_date": row_date,
            "result": result,
            "profit": "" if math.isnan(profit) else f"{profit:.2f}",
            "clv_pct": "" if math.isnan(clv_pct) else f"{clv_pct:.2f}",
            "closing_odds": "" if closing_odds is None else format_american(closing_odds),
            "closing_line": "" if closing_line is None else format_float(closing_line),
            "home_score": str(int(home_score)),
            "away_score": str(int(away_score)),
        }
    )
    return settled_row


def snapshot_open_market_row(row: dict[str, str], event: dict[str, Any]) -> dict[str, str]:
    competition = safe_get(event, "competitions", 0)
    if not competition:
        return row

    closing_market = extract_closing_market(competition, row)
    closing_odds = closing_market.get("odds")
    closing_line = closing_market.get("line")
    if closing_odds is None and closing_line is None:
        return row

    updated = dict(row)
    odds_value = parse_american(updated.get("odds"))
    clv_pct = calculate_clv_pct(odds_value, closing_odds) if odds_value is not None else math.nan
    if closing_odds is not None:
        updated["closing_odds"] = format_american(closing_odds)
    if closing_line is not None:
        updated["closing_line"] = format_float(closing_line)
    if not math.isnan(clv_pct):
        updated["clv_pct"] = f"{clv_pct:.2f}"
    return updated


def find_matching_event(events: list[dict[str, Any]], row: dict[str, str]) -> dict[str, Any] | None:
    event_name = str(row.get("event", "")).strip()
    if event_name:
        for event in events:
            if str(event.get("name", "")).strip() == event_name:
                return event

    home_team = str(row.get("home_team", "")).strip()
    away_team = str(row.get("away_team", "")).strip()
    target_matchup = matchup_key(home_team, away_team)
    if target_matchup:
        for event in events:
            competition = safe_get(event, "competitions", 0)
            if not competition:
                continue
            home = next((item for item in competition.get("competitors", []) if item.get("homeAway") == "home"), None)
            away = next((item for item in competition.get("competitors", []) if item.get("homeAway") == "away"), None)
            if not home or not away:
                continue
            event_matchup = matchup_key(str(home.get("team", {}).get("displayName", "")).strip(), str(away.get("team", {}).get("displayName", "")).strip())
            if event_matchup == target_matchup:
                return event
    return None


def normalize_tracking_row(row: dict[str, str]) -> dict[str, str]:
    updated = dict(row)
    event_id = str(updated.get("event_id", "")).strip()
    guessed_event_id = guess_event_id_from_bet_id(str(updated.get("bet_id", "")).strip())
    if (not event_id or ":" in event_id) and guessed_event_id:
        updated["event_id"] = guessed_event_id
    if "source" in updated and not str(updated.get("source", "")).strip():
        updated["source"] = "espn"
    return updated


def guess_event_id_from_bet_id(bet_id: str) -> str:
    match = re.match(r"^(oddsapi-[^-]+|\d{6,})-", bet_id)
    return match.group(1) if match else ""


def load_override_map(path: Path) -> dict[str, dict[str, str]]:
    overrides: dict[str, dict[str, str]] = {}
    for row in read_csv_rows(path):
        bet_id = str(row.get("bet_id", "")).strip()
        if bet_id:
            overrides[bet_id] = row
    return overrides


def apply_settlement_override(row: dict[str, str], override: dict[str, str]) -> dict[str, str]:
    updated = dict(row)
    result = str(override.get("result", row.get("result", ""))).strip().lower()
    stake = parse_numeric(updated.get("stake", ""))
    odds_value = parse_american(updated.get("odds"))
    manual_profit = parse_numeric(override.get("profit", ""))
    profit = manual_profit
    if math.isnan(profit) and not math.isnan(stake) and odds_value is not None and result in {"win", "loss", "push"}:
        profit = settle_profit(stake, odds_value, result)
    manual_clv = parse_numeric(override.get("clv_pct", ""))
    closing_odds = parse_american(override.get("closing_odds"))
    if math.isnan(manual_clv) and odds_value is not None:
        manual_clv = calculate_clv_pct(odds_value, closing_odds)
    updated.update(
        {
            "settled_date": normalize_iso_date(str(override.get("settled_date", ""))) if re.sub(r"[^0-9]", "", str(override.get("settled_date", ""))) else updated.get("date", ""),
            "result": result or updated.get("result", ""),
            "profit": "" if math.isnan(profit) else f"{profit:.2f}",
            "closing_odds": "" if closing_odds is None else format_american(closing_odds),
            "closing_line": format_float(parse_numeric(override.get("closing_line", ""))),
            "clv_pct": "" if math.isnan(manual_clv) else f"{manual_clv:.2f}",
            "home_score": str(override.get("home_score", updated.get("home_score", ""))).strip(),
            "away_score": str(override.get("away_score", updated.get("away_score", ""))).strip(),
        }
    )
    notes = str(updated.get("notes", "")).strip()
    override_notes = str(override.get("notes", "")).strip()
    if override_notes:
        updated["notes"] = f"{notes} Manual settlement override: {override_notes}.".strip()
    return updated


def apply_closing_override(row: dict[str, str], override: dict[str, str]) -> dict[str, str]:
    updated = dict(row)
    closing_odds = parse_american(override.get("closing_odds"))
    closing_line_text = format_float(parse_numeric(override.get("closing_line", "")))
    if closing_odds is not None:
        updated["closing_odds"] = format_american(closing_odds)
    if closing_line_text:
        updated["closing_line"] = closing_line_text

    manual_clv = parse_numeric(override.get("clv_pct", ""))
    bet_odds = parse_american(updated.get("odds"))
    if math.isnan(manual_clv) and bet_odds is not None:
        manual_clv = calculate_clv_pct(bet_odds, closing_odds)
    if not math.isnan(manual_clv):
        updated["clv_pct"] = f"{manual_clv:.2f}"

    override_notes = str(override.get("notes", "")).strip()
    notes = str(updated.get("notes", "")).strip()
    if override_notes and "Closing line override" not in notes:
        updated["notes"] = f"{notes} Closing line override: {override_notes}.".strip()
    return updated


def event_is_final(event: dict[str, Any], competition: dict[str, Any]) -> bool:
    completed = bool(
        safe_get(competition, "status", "type", "completed", default=False)
        or safe_get(event, "status", "type", "completed", default=False)
    )
    state = str(safe_get(competition, "status", "type", "state", default="")).strip().lower()
    name = str(safe_get(competition, "status", "type", "name", default="")).strip().upper()
    description = str(safe_get(competition, "status", "type", "description", default="")).strip().lower()
    return completed or state == "post" or name == "STATUS_FINAL" or description == "final"


def grade_result(row: dict[str, str], home_score: int, away_score: int) -> str | None:
    market_type = str(row.get("market_type", "")).strip().lower()
    selection = str(row.get("selection", "")).strip().lower()
    line_value = parse_numeric(row.get("line_value", ""))
    sport = str(row.get("sport", "")).strip()

    if market_type == "moneyline":
        # Check for draw in sports that allow it (soccer)
        if home_score == away_score and sport in SOCCER_SPORTS:
            return "push"
        if selection == "home":
            return "win" if home_score > away_score else "loss"
        if selection == "away":
            return "win" if away_score > home_score else "loss"
        return None

    if market_type == "spread":
        if math.isnan(line_value):
            return None
        if selection == "home":
            graded_margin = home_score + line_value - away_score
        elif selection == "away":
            graded_margin = away_score + line_value - home_score
        else:
            return None
        return score_from_margin(graded_margin)

    if market_type == "total":
        if math.isnan(line_value):
            return None
        total_score = home_score + away_score
        if selection == "over":
            return score_from_margin(total_score - line_value)
        if selection == "under":
            return score_from_margin(line_value - total_score)
        return None

    return None


def score_from_margin(value: float) -> str:
    if abs(value) < 1e-9:
        return "push"
    return "win" if value > 0 else "loss"


def settle_profit(stake: float, odds: int, result: str) -> float:
    if result == "push":
        return 0.0
    if result == "loss":
        return -stake
    return stake * american_to_profit_multiple(odds)


def extract_closing_market(competition: dict[str, Any], row: dict[str, str]) -> dict[str, float | int | None]:
    odds_block = safe_get(competition, "odds", 0, default={}) or {}
    market_type = str(row.get("market_type", "")).strip().lower()
    selection = str(row.get("selection", "")).strip().lower()

    if market_type == "moneyline" and selection in {"home", "away"}:
        return {"odds": extract_odds(odds_block.get("moneyline", {}), selection), "line": None}
    if market_type == "spread" and selection in {"home", "away"}:
        side = extract_spread_side(odds_block.get("pointSpread", {}), selection)
        return {"odds": None if not side else side["odds"], "line": None if not side else side["line"]}
    if market_type == "total" and selection in {"over", "under"}:
        side = extract_total_side(odds_block.get("total", {}), selection)
        return {"odds": None if not side else side["odds"], "line": None if not side else side["line"]}
    return {"odds": None, "line": None}


def calculate_clv_pct(bet_odds: int, closing_odds: int | None) -> float:
    if closing_odds is None:
        return math.nan
    return (american_to_probability(closing_odds) - american_to_probability(bet_odds)) * 100.0


def load_model_performance() -> dict[str, Any]:
    rows = read_csv_rows(MODEL_RESULTS_PATH)
    settled_rows = [row for row in rows if is_settled_result(row)]
    summary = summarize_performance_rows(settled_rows)
    summary["total_bets"] = len(rows)
    summary["open_bets"] = sum(1 for row in rows if not is_settled_result(row))
    summary["by_sport"] = summarize_performance_breakdown(settled_rows, "sport")
    summary["by_market"] = summarize_performance_breakdown(settled_rows, "market_type")
    return summary


def summarize_performance_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    wins = sum(1 for row in rows if str(row.get("result", "")).strip().lower() == "win")
    losses = sum(1 for row in rows if str(row.get("result", "")).strip().lower() == "loss")
    pushes = sum(1 for row in rows if str(row.get("result", "")).strip().lower() == "push")
    staked = sum(value for value in (parse_numeric(row.get("stake", "")) for row in rows) if not math.isnan(value))
    profit = sum(value for value in (parse_numeric(row.get("profit", "")) for row in rows) if not math.isnan(value))
    clv_values = [value for value in (parse_numeric(row.get("clv_pct", "")) for row in rows) if not math.isnan(value)]
    ev_values = [value for value in (parse_numeric(row.get("ev_pct", "")) for row in rows) if not math.isnan(value)]
    decisions = wins + losses
    return {
        "settled_bets": len(rows),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "staked": round(staked, 2),
        "profit": round(profit, 2),
        "roi_pct": round((profit / staked * 100.0) if staked else 0.0, 2),
        "hit_rate_pct": round((wins / decisions * 100.0) if decisions else 0.0, 2),
        "avg_clv_pct": round(sum(clv_values) / len(clv_values), 2) if clv_values else 0.0,
        "avg_ev_pct": round(sum(ev_values) / len(ev_values), 2) if ev_values else 0.0,
    }


def summarize_performance_breakdown(rows: list[dict[str, str]], key: str) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        bucket_key = str(row.get(key, "")).strip()
        if not bucket_key:
            continue
        buckets.setdefault(bucket_key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for bucket_key, bucket_rows in buckets.items():
        summary = summarize_performance_rows(bucket_rows)
        summary_rows.append(
            {
                "label": bucket_key,
                "bets": summary["settled_bets"],
                "roi_pct": summary["roi_pct"],
                "profit": summary["profit"],
                "avg_clv_pct": summary["avg_clv_pct"],
                "hit_rate_pct": summary["hit_rate_pct"],
            }
        )
    summary_rows.sort(key=lambda item: (-item["bets"], item["label"]))
    return summary_rows


def summarize_filter_reasons(candidates: list[dict[str, Any]]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for item in candidates:
        for reason in item.get("filter_reasons", []):
            counts[reason] = counts.get(reason, 0) + 1
    return sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))


def ensure_tracking_files() -> None:
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not BET_RESULTS_PATH.exists():
        BET_RESULTS_PATH.write_text(
            "bet_id,date,placed_date,settled_date,sport,event,market_type,bet_type,odds,closing_odds,stake,result,profit,clv_pct,notes\n",
            encoding="utf-8",
        )
    if not RECOMMENDATIONS_LOG_PATH.exists():
        RECOMMENDATIONS_LOG_PATH.write_text(
            "date,bet_id,sport,event,event_id,market_type,bet_type,selection,line,odds,true_probability,ev_pct,stake,stake_pct,edge_score,history_penalty_pct,notes\n",
            encoding="utf-8",
        )
    if not MODEL_RESULTS_PATH.exists():
        MODEL_RESULTS_PATH.write_text(
            "bet_id,date,placed_date,settled_date,sport,event,event_id,market_type,source,bet_type,selection,selection_team,line,line_value,home_team,away_team,odds,closing_odds,closing_line,stake,true_probability,ev_pct,edge_score,history_penalty_pct,sportsbook,result,profit,clv_pct,home_score,away_score,notes\n",
            encoding="utf-8",
        )
    if not CLOSING_LINE_OVERRIDES_PATH.exists():
        CLOSING_LINE_OVERRIDES_PATH.write_text(
            "bet_id,closing_odds,closing_line,clv_pct,notes\n",
            encoding="utf-8",
        )
    if not SETTLEMENT_OVERRIDES_PATH.exists():
        SETTLEMENT_OVERRIDES_PATH.write_text(
            "bet_id,settled_date,result,profit,closing_odds,closing_line,clv_pct,home_score,away_score,notes\n",
            encoding="utf-8",
        )


def fetch_json_value(url: str) -> Any:
    payload, _ = fetch_json_with_headers(url)
    return payload


def fetch_json_with_headers(url: str) -> tuple[Any, dict[str, Any]]:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8")), dict(response.headers)


def fetch_json(url: str) -> Any:
    return fetch_json_value(url)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8").lstrip("\ufeff"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def safe_get(obj: Any, *path: Any, default: Any = None) -> Any:
    current = obj
    for key in path:
        try:
            if isinstance(current, list):
                current = current[key]
            else:
                current = current[key]
        except (KeyError, IndexError, TypeError):
            return default
    return current


def parse_record(summary: str) -> dict[str, float]:
    parts = [int(piece) for piece in re.findall(r"\d+", summary or "")]
    if not parts:
        return {"games": 0.0, "win_pct": 0.5, "points_pct": 0.5}
    games = float(sum(parts))
    if len(parts) == 2:
        wins, losses = parts
        return {
            "games": games,
            "win_pct": wins / games if games else 0.5,
            "points_pct": wins / games if games else 0.5,
        }
    wins, losses, extra = parts[0], parts[1], parts[2]
    return {
        "games": games,
        "win_pct": wins / games if games else 0.5,
        "points_pct": ((wins * 2) + extra) / (games * 2) if games else 0.5,
    }


def parse_soccer_record(summary: str) -> dict[str, float]:
    parts = [int(piece) for piece in re.findall(r"\d+", summary or "")]
    if not parts:
        return {"games": 0.0, "win_pct": 0.5, "points_pct": 0.5}
    if len(parts) >= 3:
        wins, losses, draws = parts[0], parts[1], parts[2]
        games = float(wins + losses + draws)
        return {
            "games": games,
            "win_pct": wins / games if games else 0.5,
            "points_pct": ((wins * 3) + draws) / (games * 3) if games else 0.5,
        }
    return parse_record(summary)


def parse_numeric(raw: Any) -> float:
    if raw is None:
        return math.nan
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip()
    if not text:
        return math.nan
    text = text.replace("%", "").replace(",", "")
    match = re.search(r"[-+]?\d*\.?\d+", text)
    return float(match.group(0)) if match else math.nan


def safe_int(raw: Any, default: int = 0) -> int:
    value = parse_numeric(raw)
    if math.isnan(value):
        return default
    return int(value)


def parse_probability(raw: Any) -> float | None:
    value = parse_numeric(raw)
    if math.isnan(value):
        return None
    if value > 1.0:
        value /= 100.0
    return clamp(value, 0.01, 0.99)


def parse_odds_to_american(raw: Any, odds_format: str = "") -> int | None:
    if raw is None or str(raw).strip() == "":
        return None
    fmt = odds_format.strip().lower()
    if fmt in {"american", "us"}:
        return parse_american(raw)
    if fmt in {"decimal", "eu"}:
        return decimal_to_american(raw)

    text = str(raw).strip()
    if text.startswith("+") or text.startswith("-"):
        return parse_american(raw)

    value = parse_numeric(raw)
    if math.isnan(value):
        return None
    if 1.01 <= value < 10.0:
        return decimal_to_american(value)
    return parse_american(raw)


def parse_american(raw: Any) -> int | None:
    value = parse_numeric(raw)
    if math.isnan(value):
        return None
    number = int(round(value))
    if number == 0:
        return None
    return number


def decimal_to_american(raw: Any) -> int | None:
    value = parse_numeric(raw)
    if math.isnan(value) or value <= 1.0:
        return None
    if value >= 2.0:
        return int(round((value - 1.0) * 100.0))
    return int(round(-100.0 / (value - 1.0)))


def parse_probable_metric(record: str, fallback: float) -> float:
    values = re.findall(r"[-+]?\d*\.?\d+", record or "")
    if not values:
        return fallback
    return float(values[-1])


def parse_line_value(raw: str | None) -> float | None:
    value = parse_numeric(raw)
    return None if math.isnan(value) else value


def external_feed_quality(row: dict[str, Any], model_context: dict[str, Any] | None) -> float:
    reference_books = int(parse_numeric(row.get("reference_book_count", "")) if not math.isnan(parse_numeric(row.get("reference_book_count", ""))) else 0)
    sharp_books = int(parse_numeric(row.get("sharp_book_count", "")) if not math.isnan(parse_numeric(row.get("sharp_book_count", ""))) else 0)
    quality = 54.0 + (reference_books * 3.0) + (sharp_books * 2.5)
    if model_context:
        quality = max(quality, float(model_context["model"]["quality"]) * 0.92)
    return clamp(quality, 48.0, 92.0)


def external_feed_sample_size(row: dict[str, Any], model_context: dict[str, Any] | None) -> int:
    if model_context:
        return max(18, int(model_context["sample_size"]))
    reference_books = int(parse_numeric(row.get("reference_book_count", "")) if not math.isnan(parse_numeric(row.get("reference_book_count", ""))) else 0)
    sharp_books = int(parse_numeric(row.get("sharp_book_count", "")) if not math.isnan(parse_numeric(row.get("sharp_book_count", ""))) else 0)
    return max(18, (reference_books * 6) + (sharp_books * 2))


def consensus_quality(consensus: dict[str, Any]) -> float:
    return clamp(54.0 + (consensus.get("books", 0) * 3.0) + (consensus.get("sharp_books", 0) * 2.5), 48.0, 88.0)


def consensus_sample_size(consensus: dict[str, Any]) -> int:
    return max(18, int(consensus.get("books", 0)) * 6 + int(consensus.get("sharp_books", 0)) * 2)


def blend_external_model_probability(
    *,
    consensus_probability: float | None,
    sharp_probability: float | None,
    derived_probability: float | None,
    model_context: dict[str, Any] | None,
    config: dict[str, Any],
) -> float | None:
    provider_cfg = config.get("provider_feed", {})
    market_probability = None
    if sharp_probability is not None and consensus_probability is not None:
        sharp_weight = float(provider_cfg.get("sharp_consensus_weight", 0.65))
        market_probability = (sharp_probability * sharp_weight) + (consensus_probability * (1.0 - sharp_weight))
    elif sharp_probability is not None:
        market_probability = sharp_probability
    elif consensus_probability is not None:
        market_probability = consensus_probability

    if derived_probability is not None and market_probability is not None:
        internal_weight = float(provider_cfg.get("internal_model_weight", 0.55))
        consensus_weight = float(provider_cfg.get("consensus_model_weight", 0.45))
        if model_context and model_context["model"]["quality"] < 66.0:
            internal_weight *= 0.85
        total_weight = max(internal_weight + consensus_weight, 0.01)
        return clamp(((derived_probability * internal_weight) + (market_probability * consensus_weight)) / total_weight, 0.01, 0.99)

    if market_probability is not None:
        return clamp(market_probability, 0.01, 0.99)
    if derived_probability is not None:
        return clamp(derived_probability, 0.01, 0.99)
    return None


def build_candidate_feature_vector(
    *,
    sport: str,
    market_type: str,
    sample_size: int,
    quality: float,
    fair_probability: float,
    model_probability: float,
    odds: int,
    opposite_odds: int | None,
    line_value: float | None,
    home: dict[str, Any] | None,
    away: dict[str, Any] | None,
    model: dict[str, Any] | None,
) -> list[float]:
    features = [
        clamp(sample_size / 100.0, 0.0, 1.5),
        clamp(quality / 100.0, 0.0, 1.0),
        clamp(fair_probability, 0.01, 0.99),
        clamp(model_probability, 0.01, 0.99),
        american_to_probability(odds),
        0.5 if opposite_odds is None else american_to_probability(opposite_odds),
        american_to_profit_multiple(odds),
        0.0 if line_value is None or math.isnan(line_value) else line_value,
        0.0 if model is None else float(model.get("margin", 0.0)),
        0.0 if model is None else float(model.get("expected_total", 0.0)),
        0.0 if model is None else float(model.get("quality", quality)) / 100.0,
    ]

    if home and away:
        features.extend(
            [
                float(home.get("win_pct", 0.5)) - float(away.get("win_pct", 0.5)),
                float(home.get("home_pct", 0.5)) - float(away.get("road_pct", 0.5)),
                float(home.get("last_10_win_pct", home.get("win_pct", 0.5))) - float(away.get("last_10_win_pct", away.get("win_pct", 0.5))),
                float(home.get("last_5_win_pct", home.get("win_pct", 0.5))) - float(away.get("last_5_win_pct", away.get("win_pct", 0.5))),
            ]
        )
        if sport in BASEBALL_SPORTS:
            features.extend(
                [
                    float(home.get("runs_pg", 0.0)) - float(away.get("runs_pg", 0.0)),
                    float(away.get("team_era", 0.0)) - float(home.get("team_era", 0.0)),
                    float(away.get("starter_era", 0.0)) - float(home.get("starter_era", 0.0)),
                ]
            )
        elif sport in BASKETBALL_SPORTS:
            features.extend(
                [
                    float(home.get("avg_points", 0.0)) - float(away.get("avg_points", 0.0)),
                    float(home.get("fg_pct", 0.0)) - float(away.get("fg_pct", 0.0)),
                    float(home.get("three_pct", 0.0)) - float(away.get("three_pct", 0.0)),
                    float(home.get("avg_rebounds", 0.0)) - float(away.get("avg_rebounds", 0.0)),
                ]
            )
        elif sport in HOCKEY_SPORTS:
            features.extend(
                [
                    float(home.get("goals_pg", 0.0)) - float(away.get("goals_pg", 0.0)),
                    float(home.get("save_pct", 0.9)) - float(away.get("save_pct", 0.9)),
                    float(home.get("points_pct", 0.5)) - float(away.get("points_pct", 0.5)),
                ]
            )
        elif sport in FOOTBALL_SPORTS:
            features.extend(
                [
                    float(home.get("points_pct", 0.5)) - float(away.get("points_pct", 0.5)),
                    float(home.get("home_pct", 0.5)) - float(away.get("road_pct", 0.5)),
                ]
            )
        elif sport in SOCCER_SPORTS:
            features.extend(
                [
                    float(home.get("goals_pg", 0.0)) - float(away.get("goals_pg", 0.0)),
                    float(home.get("shots_on_target_pg", 0.0)) - float(away.get("shots_on_target_pg", 0.0)),
                    float(home.get("possession_pct", 50.0)) - float(away.get("possession_pct", 50.0)),
                    float(home.get("points_pct", 0.5)) - float(away.get("points_pct", 0.5)),
                ]
            )

    features.append(1.0 if market_type == "moneyline" else 0.0)
    features.append(1.0 if market_type == "spread" else 0.0)
    features.append(1.0 if market_type == "total" else 0.0)
    return [round(float(value), 6) for value in features]


def quality_score(sport: str, home: dict[str, Any], away: dict[str, Any], keys: list[str], stats_defaulted: bool = False) -> float:
    present = 0
    for key in keys:
        if not math.isnan(float(home.get(key, math.nan))) and not math.isnan(float(away.get(key, math.nan))):
            present += 1
    base_map = {
        "MLB": 76.0,
        "NCAABASE": 68.0,
        "NBA": 70.0,
        "WNBA": 64.0,
        "NCAAMB": 66.0,
        "NCAAWB": 65.0,
        "NFL": 60.0,
        "NCAAF": 58.0,
        "NHL": 72.0,
        "MLS": 64.0,
        "EPL": 66.0,
        "La Liga": 65.0,
        "Bundesliga": 67.0,
        "Serie A": 64.0,
        "Ligue 1": 63.0,
        "UCL": 68.0,
    }
    base = base_map.get(sport, 60.0)
    completeness = (present / max(len(keys), 1)) * 18.0
    sample_bonus = min(home["games"], away["games"]) * 0.25
    quality = base + completeness + sample_bonus
    # Reduce quality score if stats were defaulted
    if stats_defaulted:
        quality -= 8.0  # Penalty for using defaulted stats
    return clamp(quality, 45.0, 95.0)


def american_to_probability(odds: int) -> float:
    return 100.0 / (odds + 100.0) if odds > 0 else abs(odds) / (abs(odds) + 100.0)


def american_to_profit_multiple(odds: int) -> float:
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def probability_to_american(probability: float) -> int:
    probability = clamp(probability, 0.01, 0.99)
    if probability >= 0.5:
        return round((-100.0 * probability) / (1.0 - probability))
    return round((100.0 * (1.0 - probability)) / probability)


def devig_probability(odds_a: int, odds_b: int) -> float:
    prob_a = american_to_probability(odds_a)
    prob_b = american_to_probability(odds_b)
    total = prob_a + prob_b
    return prob_a / total if total else prob_a


def logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


class MLModelManager:
    """Manages candidate-level ML models for bet outcome prediction."""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.feature_shapes = {}

    def save_models(self, path: Path = ML_MODELS_PATH) -> None:
        try:
            model_data = {
                "models": self.models,
                "scalers": self.scalers,
                "feature_importance": self.feature_importance,
                "feature_shapes": self.feature_shapes,
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            import joblib
            joblib.dump(model_data, path)
            print(f"Saved ML models to {path}")
        except Exception as e:
            print(f"Error saving ML models: {e}")

    def load_models(self, path: Path = ML_MODELS_PATH) -> bool:
        try:
            if not path.exists():
                return False

            import joblib
            model_data = joblib.load(path)

            self.models = model_data.get("models", {})
            self.scalers = model_data.get("scalers", {})
            self.feature_importance = model_data.get("feature_importance", {})
            self.feature_shapes = model_data.get("feature_shapes", {})

            print(f"Loaded ML models for {len(self.models)} model buckets from {path}")
            return True
        except Exception as e:
            print(f"Error loading ML models: {e}")
            return False

    def train_model(self, model_key: str, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        if len(X) < 20:
            return {"success": False, "error": "Not enough training data"}
        if len(set(y.tolist())) < 2:
            return {"success": False, "error": "Need both win and loss examples"}

        class_counts = np.bincount(y.astype(int))
        if len(class_counts) < 2 or min(class_counts) < 4:
            return {"success": False, "error": "Class balance too thin"}

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        )
        rf_model.fit(X_train_scaled, y_train)

        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )
        gb_model.fit(X_train_scaled, y_train)

        rf_pred = rf_model.predict_proba(X_test_scaled)[:, 1]
        gb_pred = gb_model.predict_proba(X_test_scaled)[:, 1]
        rf_auc = roc_auc_score(y_test, rf_pred) if len(set(y_test)) > 1 else 0.5
        gb_auc = roc_auc_score(y_test, gb_pred) if len(set(y_test)) > 1 else 0.5

        self.models[model_key] = {
            "random_forest": rf_model,
            "gradient_boosting": gb_model,
        }
        self.scalers[model_key] = scaler
        self.feature_importance[model_key] = {
            "random_forest": rf_model.feature_importances_.tolist(),
            "gradient_boosting": gb_model.feature_importances_.tolist(),
        }
        self.feature_shapes[model_key] = int(X.shape[1])

        return {
            "success": True,
            "rf_auc": rf_auc,
            "gb_auc": gb_auc,
            "ensemble_auc": (rf_auc + gb_auc) / 2,
        }

    def predict_market(self, sport: str, market_type: str, features: list[float] | None) -> float | None:
        if not features:
            return None
        model_key = f"{sport}:{market_type}"
        if model_key not in self.models or model_key not in self.scalers:
            return None
        expected_shape = int(self.feature_shapes.get(model_key, 0))
        if expected_shape and len(features) != expected_shape:
            return None

        feature_array = np.array(features, dtype=float).reshape(1, -1)
        features_scaled = self.scalers[model_key].transform(feature_array)
        rf_prob = self.models[model_key]["random_forest"].predict_proba(features_scaled)[0, 1]
        gb_prob = self.models[model_key]["gradient_boosting"].predict_proba(features_scaled)[0, 1]
        ensemble_prob = (rf_prob * 0.5) + (gb_prob * 0.5)
        return clamp(float(ensemble_prob), 0.01, 0.99)


# Global ML model manager
ml_manager = MLModelManager()
# Load persisted models on startup
ml_manager.load_models()


def store_historical_data(candidates: list[dict[str, Any]], day_iso: str) -> None:
    """Store daily candidate data for ML training"""
    try:
        # Load existing historical data
        historical_data = {}
        if HISTORICAL_DATA_PATH.exists():
            historical_data = json.loads(HISTORICAL_DATA_PATH.read_text(encoding="utf-8"))
        
        # Store today's data
        historical_data[day_iso] = {
            "candidates": candidates,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Save historical data
        HISTORICAL_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        HISTORICAL_DATA_PATH.write_text(json.dumps(historical_data, indent=2), encoding="utf-8")
        
        print(f"Stored {len(candidates)} historical data points for {day_iso}")
    except Exception as e:
        print(f"Error storing historical data: {e}")


def load_historical_data() -> dict[str, Any]:
    """Load all historical data for ML training"""
    try:
        if not HISTORICAL_DATA_PATH.exists():
            return {}
        
        historical_data = json.loads(HISTORICAL_DATA_PATH.read_text(encoding="utf-8"))
        return historical_data
    except Exception as e:
        print(f"Error loading historical data: {e}")
        return {}


def load_model_results_map() -> dict[str, dict[str, str]]:
    """Load model results as a map by bet_id for training labels"""
    results_map = {}
    if not MODEL_RESULTS_PATH.exists():
        return results_map
    
    try:
        for row in read_csv_rows(MODEL_RESULTS_PATH):
            bet_id = row.get("bet_id", "")
            result = row.get("result", "").lower()
            if bet_id and result in ("win", "loss", "push"):
                results_map[bet_id] = {"result": result, "row": row}
    except Exception as e:
        print(f"Error loading model results: {e}")
    
    return results_map


def prepare_training_data(historical_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Prepare candidate-level training data from historical snapshots."""
    training_data = {}
    model_results = load_model_results_map()
    print(f"Loaded {len(model_results)} settled results for training")

    for day_iso, day_data in historical_data.items():
        for candidate in day_data.get("candidates", []):
            sport = candidate.get("sport")
            market_type = candidate.get("marketType") or candidate.get("market_type")
            if not sport or not market_type:
                continue

            model_key = f"{sport}:{str(market_type).strip().lower()}"
            if model_key not in training_data:
                training_data[model_key] = {"X": [], "y": []}

            try:
                bet_id = candidate.get("id", "")
                result_data = model_results.get(bet_id)
                if not result_data:
                    continue

                result = result_data["result"]
                if result == "push":
                    continue
                label = 1 if result == "win" else 0

                features = candidate.get("training_features", [])
                if not isinstance(features, list) or not features:
                    features = [
                        candidate.get("dataQuality", 50) / 100.0,
                        candidate.get("sampleSize", 0) / 100.0,
                        parse_probability(candidate.get("fair_probability_pct")) or 0.5,
                        parse_probability(candidate.get("true_probability_pct")) or 0.5,
                    ]

                training_data[model_key]["X"].append([float(value) for value in features])
                training_data[model_key]["y"].append(label)
            except Exception:
                continue

    for model_key, data in training_data.items():
        print(f"{model_key}: {len(data['X'])} samples, {sum(data['y'])} wins")

    return training_data


def train_ml_models_from_historical_data() -> dict[str, Any]:
    historical_data = load_historical_data()
    if not historical_data:
        print("No historical data available for training")
        return {"success": False, "error": "No historical data"}
    
    print(f"Loading historical data from {len(historical_data)} days")
    
    # Prepare training data
    training_data = prepare_training_data(historical_data)
    
    results = {}
    for model_key, data in training_data.items():
        if len(data["X"]) < 20:
            print(f"Not enough data for {model_key} ({len(data['X'])} samples)")
            continue

        X = np.array(data["X"])
        y = np.array(data["y"])

        print(f"Training ML model for {model_key} with {len(X)} samples")
        result = ml_manager.train_model(model_key, X, y)
        results[model_key] = result

        if result.get("success"):
            print(f"  {model_key} RF AUC: {result.get('rf_auc', 0):.3f}")
            print(f"  {model_key} GB AUC: {result.get('gb_auc', 0):.3f}")
            print(f"  {model_key} Ensemble AUC: {result.get('ensemble_auc', 0):.3f}")

    if ml_manager.models:
        ml_manager.save_models()

    return {"success": True, "results": results}


def backtest_model(historical_data: dict[str, Any], sport: str) -> dict[str, Any]:
    """Backtest model on historical data"""
    if sport not in ml_manager.models:
        return {"success": False, "error": "Model not trained"}
    
    # Get data for this sport
    sport_data = []
    for day_iso, day_data in historical_data.items():
        for candidate in day_data.get("candidates", []):
            if candidate.get("sport") == sport:
                sport_data.append(candidate)
    
    if len(sport_data) < 10:
        return {"success": False, "error": "Not enough data for backtesting"}
    
    # Simulate predictions and calculate metrics
    correct_predictions = 0
    total_predictions = 0
    total_edge = 0.0
    
    for candidate in sport_data:
        # Get model prediction (simplified - would need full team data in production)
        # For now, use the edge_pct as a proxy
        edge = candidate.get("edge_pct", 0)
        total_edge += edge
        total_predictions += 1
        
        # Count as correct if edge > 0 (positive expected value)
        if edge > 0:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    avg_edge = total_edge / total_predictions if total_predictions > 0 else 0
    
    return {
        "success": True,
        "total_predictions": total_predictions,
        "correct_predictions": correct_predictions,
        "accuracy": accuracy,
        "avg_edge_pct": avg_edge,
    }


def linear_regression(x_values: list[float], y_values: list[float]) -> tuple[float, float]:
    """Simple linear regression: returns (slope, intercept)"""
    n = len(x_values)
    if n < 2:
        return 0.0, 0.0
    
    sum_x = sum(x_values)
    sum_y = sum(y_values)
    sum_xy = sum(x * y for x, y in zip(x_values, y_values))
    sum_x2 = sum(x * x for x in x_values)
    
    denominator = n * sum_x2 - sum_x * sum_x
    if abs(denominator) < 1e-10:
        return 0.0, 0.0
    
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    
    return slope, intercept


def regression_predict(slope: float, intercept: float, x: float) -> float:
    """Predict y value using linear regression"""
    return slope * x + intercept


def weighted_blend(prior: float, likelihood: float, weight: float = 0.5) -> float:
    """Blend two probabilities using a weighted average"""
    # Combine prior belief with new evidence using weighted average
    return clamp(prior * (1 - weight) + likelihood * weight, 0.01, 0.99)


def poisson_probability(k: float, lambda_param: float) -> float:
    """Calculate Poisson probability for k events with rate lambda_param"""
    if k < 0 or lambda_param <= 0:
        return 0.0
    # Use approximation for large k
    if k > 100:
        return 0.0
    try:
        # P(k; λ) = (λ^k * e^(-λ)) / k!
        log_prob = k * math.log(lambda_param) - lambda_param - sum(math.log(i) for i in range(1, int(k) + 1))
        return min(math.exp(log_prob), 1.0)
    except (OverflowError, ValueError):
        return 0.0


def normal_cdf(x: float, mean: float, sd: float) -> float:
    z = (x - mean) / (sd * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def spread_cover_probability(selection: str, line_value: float, margin: float, sd: float) -> float:
    if selection == "home":
        return 1.0 - normal_cdf(-line_value, margin, sd)
    return normal_cdf(line_value, margin, sd)


def points_match(value_a: float, value_b: float, tolerance: float = 0.001) -> bool:
    return abs(value_a - value_b) <= tolerance


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def matchup_key(home_team: str, away_team: str) -> str:
    home_slug = slugify(home_team)
    away_slug = slugify(away_team)
    if not home_slug or not away_slug:
        return ""
    return f"{away_slug}-at-{home_slug}"


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def format_american(odds: int) -> str:
    return f"+{odds}" if odds > 0 else str(odds)


def format_float(value: float) -> str:
    if math.isnan(value):
        return ""
    rounded = round(value, 3)
    if abs(rounded - round(rounded)) < 1e-9:
        return str(int(round(rounded)))
    return f"{rounded:.3f}".rstrip("0").rstrip(".")


def html_escape(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def normalize_iso_date(raw: str) -> str:
    digits = re.sub(r"[^0-9]", "", raw)
    if len(digits) >= 8:
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    return date.today().isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
