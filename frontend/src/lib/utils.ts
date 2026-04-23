import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatPercent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`
}

export function formatOdds(price: number): string {
  return price.toFixed(2)
}

export function formatEdge(edge: number): string {
  const sign = edge >= 0 ? '+' : ''
  return `${sign}${(edge * 100).toFixed(2)}%`
}

export function marketLabel(market: string, selection: string, line: number | null): string {
  if (market === '1x2') {
    return { home: '1 (Home)', draw: 'X (Draw)', away: '2 (Away)' }[selection] ?? selection
  }
  if (market === 'double_chance') {
    return { '1x': '1X', '12': '12', x2: 'X2' }[selection] ?? selection
  }
  if (market === 'over_under') {
    return `${selection === 'over' ? 'Over' : 'Under'} ${line ?? ''}`.trim()
  }
  if (market === 'btts') {
    return selection === 'yes' ? 'BTTS · Yes' : 'BTTS · No'
  }
  if (market === 'match_winner') {
    return selection === 'home' ? 'Player 1' : 'Player 2'
  }
  if (market === 'correct_score') {
    return `CS ${selection}`
  }
  if (market === 'htft') {
    const [ht, ft] = selection.split('_')
    const code = (s: string) => (s === 'home' ? '1' : s === 'away' ? '2' : 'X')
    return `HT/FT ${code(ht)}/${code(ft)}`
  }
  return `${market} · ${selection}${line != null ? ` ${line}` : ''}`
}

export function marketGroup(market: string): string {
  switch (market) {
    case '1x2':
    case 'double_chance':
      return 'Match result'
    case 'over_under':
      return 'Goals (Over / Under)'
    case 'btts':
      return 'Both teams to score'
    case 'correct_score':
      return 'Correct score'
    case 'htft':
      return 'Half-time / Full-time'
    case 'match_winner':
      return 'Match winner'
    default:
      return market
  }
}
