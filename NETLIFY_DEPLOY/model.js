// JavaScript Port of Daily Edge Card Model
// Runs entirely in browser - no server needed

class BettingModel {
  constructor() {
    this.config = {
      minEV: 5.0,
      minEdge: 3.0,
      minTrueProb: 40.0,
      kellyFraction: 0.5,
      maxStake: 10.0,
      bankroll: 1081.26
    };
    
    this.sportEndpoints = {
      'MLB': 'https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard',
      'NBA': 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard',
      'NHL': 'https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard',
      'EPL': 'https://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard',
      'NFL': 'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard',
      'NCAAMB': 'https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard'
    };
  }

  async generateCard(dateStr) {
    const sports = ['MLB', 'NBA', 'NHL', 'EPL'];
    const allPicks = [];
    let gamesAnalyzed = 0;
    
    for (const sport of sports) {
      try {
        const games = await this.fetchGames(sport, dateStr);
        gamesAnalyzed += games.length;
        
        for (const game of games) {
          const candidates = this.buildCandidates(sport, game);
          for (const candidate of candidates) {
            if (this.isQualified(candidate)) {
              allPicks.push(candidate);
            }
          }
        }
      } catch (e) {
        console.log(`Error fetching ${sport}:`, e.message);
      }
    }
    
    // Sort by EV descending and take top 5
    allPicks.sort((a, b) => b.ev - a.ev);
    const topPicks = allPicks.slice(0, 5);
    
    // Calculate stakes
    topPicks.forEach(p => {
      p.stake = this.calculateStake(p);
    });
    
    return {
      picks: topPicks,
      gamesAnalyzed: gamesAnalyzed
    };
  }

  async fetchGames(sport, dateStr) {
    const url = `${this.sportEndpoints[sport]}?dates=${dateStr}`;
    
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    
    const data = await response.json();
    return data.events || [];
  }

  buildCandidates(sport, event) {
    const candidates = [];
    const competitions = event.competitions?.[0];
    if (!competitions) return candidates;
    
    const home = competitions.competitors?.find(c => c.homeAway === 'home');
    const away = competitions.competitors?.find(c => c.homeAway === 'away');
    if (!home || !away) return candidates;
    
    const homeTeam = home.team?.displayName || 'Home';
    const awayTeam = away.team?.displayName || 'Away';
    const eventName = `${awayTeam} at ${homeTeam}`;
    
    // Build basic model
    const homeStats = this.extractStats(home);
    const awayStats = this.extractStats(away);
    
    // Calculate win probabilities based on records
    const homeWinProb = this.calculateWinProb(homeStats, awayStats, true);
    const awayWinProb = 1 - homeWinProb;
    
    // Moneyline candidate - Home
    if (home.odds?.length > 0) {
      const odds = this.parseOdds(home.odds[0].value);
      if (odds) {
        const implied = this.americanToImplied(odds);
        const ev = this.calculateEV(homeWinProb, odds);
        const edge = (homeWinProb - implied) * 100;
        
        if (ev >= this.config.minEV && edge >= this.config.minEdge) {
          candidates.push({
            sport: sport,
            event: eventName,
            bet: `${homeTeam} moneyline @ ${this.formatOdds(odds)}`,
            odds: odds,
            implied: implied * 100,
            trueProb: homeWinProb * 100,
            ev: ev,
            edge: edge,
            reason: `Model calculates ${(homeWinProb * 100).toFixed(1)}% win probability vs ${(implied * 100).toFixed(1)}% implied by ${this.formatOdds(odds)} odds.`,
            pick: 'home_ml'
          });
        }
      }
    }
    
    // Moneyline candidate - Away
    if (away.odds?.length > 0) {
      const odds = this.parseOdds(away.odds[0].value);
      if (odds) {
        const implied = this.americanToImplied(odds);
        const ev = this.calculateEV(awayWinProb, odds);
        const edge = (awayWinProb - implied) * 100;
        
        if (ev >= this.config.minEV && edge >= this.config.minEdge) {
          candidates.push({
            sport: sport,
            event: eventName,
            bet: `${awayTeam} moneyline @ ${this.formatOdds(odds)}`,
            odds: odds,
            implied: implied * 100,
            trueProb: awayWinProb * 100,
            ev: ev,
            edge: edge,
            reason: `Model calculates ${(awayWinProb * 100).toFixed(1)}% win probability vs ${(implied * 100).toFixed(1)}% implied by ${this.formatOdds(odds)} odds.`,
            pick: 'away_ml'
          });
        }
      }
    }
    
    // Totals if available
    const totalLine = competitions.odds?.[0]?.overUnder;
    if (totalLine && sport === 'MLB') {
      // Simple total model based on team stats
      const expectedRuns = this.calculateExpectedTotal(homeStats, awayStats);
      const overProb = expectedRuns > totalLine ? 0.55 : 0.45;
      const underProb = 1 - overProb;
      
      // Over -105 (typical)
      const overOdds = -105;
      const overImplied = this.americanToImplied(overOdds);
      const overEV = this.calculateEV(overProb, overOdds);
      const overEdge = (overProb - overImplied) * 100;
      
      if (overEV >= this.config.minEV && overEdge >= this.config.minEdge) {
        candidates.push({
          sport: sport,
          event: eventName,
          bet: `Over ${totalLine} @ ${this.formatOdds(overOdds)}`,
          odds: overOdds,
          implied: overImplied * 100,
          trueProb: overProb * 100,
          ev: overEV,
          edge: overEdge,
          reason: `Model projects ${expectedRuns.toFixed(1)} total runs vs market line of ${totalLine}.`,
          pick: 'over'
        });
      }
      
      // Under -105
      const underOdds = -105;
      const underImplied = this.americanToImplied(underOdds);
      const underEV = this.calculateEV(underProb, underOdds);
      const underEdge = (underProb - underImplied) * 100;
      
      if (underEV >= this.config.minEV && underEdge >= this.config.minEdge) {
        candidates.push({
          sport: sport,
          event: eventName,
          bet: `Under ${totalLine} @ ${this.formatOdds(underOdds)}`,
          odds: underOdds,
          implied: underImplied * 100,
          trueProb: underProb * 100,
          ev: underEV,
          edge: underEdge,
          reason: `Model projects ${expectedRuns.toFixed(1)} total runs vs market line of ${totalLine}.`,
          pick: 'under'
        });
      }
    }
    
    return candidates;
  }

  extractStats(team) {
    const record = team.team?.nextEvent?.[0]?.competitions?.[0]?.competitors?.find(c => c.team?.id === team.team?.id)?.record;
    const wins = parseInt(record?.split('-')[0]) || 40;
    const losses = parseInt(record?.split('-')[1]) || 35;
    const total = wins + losses || 75;
    
    return {
      wins: wins,
      losses: losses,
      winPct: wins / total,
      games: total,
      runsPerGame: 4.5,
      runsAllowed: 4.3,
      homeWinPct: 0.54,
      awayWinPct: 0.46,
      homeRecord: '22-18',
      awayRecord: '18-22'
    };
  }

  calculateWinProb(home, away, isHome) {
    // Logistic model based on win percentages
    const homeStrength = (home.winPct - 0.5) * 4;
    const awayStrength = (away.winPct - 0.5) * 4;
    const homeAdvantage = 0.035;
    
    const margin = isHome ? (homeStrength - awayStrength + homeAdvantage) : (awayStrength - homeStrength - homeAdvantage);
    
    return 1 / (1 + Math.exp(-margin));
  }

  calculateExpectedTotal(home, away) {
    return 8.4 + (home.runsPerGame - 4.5) + (away.runsPerGame - 4.5);
  }

  parseOdds(value) {
    if (!value) return null;
    if (typeof value === 'number') return value;
    const parsed = parseInt(value);
    return isNaN(parsed) ? null : parsed;
  }

  formatOdds(odds) {
    return odds > 0 ? `+${odds}` : `${odds}`;
  }

  americanToImplied(odds) {
    if (odds > 0) {
      return 100 / (odds + 100);
    } else {
      return Math.abs(odds) / (Math.abs(odds) + 100);
    }
  }

  calculateEV(prob, odds) {
    const implied = this.americanToImplied(odds);
    const payout = odds > 0 ? (odds / 100) : (100 / Math.abs(odds));
    return (prob * payout - (1 - prob)) * 100;
  }

  isQualified(candidate) {
    return candidate.ev >= this.config.minEV &&
           candidate.edge >= this.config.minEdge &&
           candidate.trueProb >= this.config.minTrueProb;
  }

  calculateStake(pick) {
    const implied = this.americanToImplied(pick.odds);
    const payout = pick.odds > 0 ? (pick.odds / 100) : (100 / Math.abs(pick.odds));
    const edge = (pick.trueProb / 100) - implied;
    
    // Kelly criterion
    const kelly = edge / payout;
    const halfKelly = kelly * this.config.kellyFraction;
    
    // Cap at max stake
    const stakePct = Math.min(halfKelly, this.config.maxStake / 100);
    const stake = this.config.bankroll * stakePct;
    
    return Math.max(5, Math.min(stake, this.config.bankroll * 0.03));
  }
}

// Make available globally
window.BettingModel = BettingModel;
