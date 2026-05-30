import { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import { useUIStore } from '@/store/uiStore';

export interface EquityPoint {
  date: string;
  strategy: number;
  benchmark: number;
}

interface EquityCurveChartProps {
  data: EquityPoint[];
}

export function EquityCurveChart({ data }: EquityCurveChartProps) {
  const theme = useUIStore((s) => s.theme);
  const isDark = theme === 'dark' || (theme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);

  const colors = {
    strategy: '#0ea5e9',
    benchmark: '#94a3b8',
    grid: isDark ? '#334155' : '#e2e8f0',
    text: isDark ? '#94a3b8' : '#64748b',
  };

  const fmtCurrency = (v: number) =>
    v >= 1e6 ? `$${(v / 1e6).toFixed(2)}M` : `$${v.toFixed(0)}`;

  return (
    <Box>
      <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
        Equity Curve
      </Typography>
      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={data} margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={colors.grid} />
          <XAxis
            dataKey="date"
            stroke={colors.text}
            tick={{ fontSize: 11 }}
            tickLine={false}
          />
          <YAxis
            stroke={colors.text}
            tick={{ fontSize: 11 }}
            tickLine={false}
            tickFormatter={fmtCurrency}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: isDark ? '#1e293b' : '#fff',
              border: `1px solid ${colors.grid}`,
              borderRadius: 8,
              fontSize: 12,
            }}
            formatter={(value: number) => [fmtCurrency(value)]}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="strategy"
            stroke={colors.strategy}
            strokeWidth={2}
            dot={false}
            name="Strategy"
          />
          <Line
            type="monotone"
            dataKey="benchmark"
            stroke={colors.benchmark}
            strokeWidth={1.5}
            strokeDasharray="5 5"
            dot={false}
            name="Benchmark"
          />
        </LineChart>
      </ResponsiveContainer>
    </Box>
  );
}

/** Generate mock equity curve data for demo purposes */
export function generateMockEquityCurve(
  startDate: string,
  endDate: string,
  initialCapital: number,
  seed = 42
): EquityPoint[] {
  const start = new Date(startDate);
  const end = new Date(endDate);
  const data: EquityPoint[] = [];

  let strategyNAV = initialCapital;
  let benchmarkNAV = initialCapital;

  // Simple pseudo-random walk with seed
  let rng = seed;
  const nextRand = () => {
    rng = (rng * 16807 + 0) % 2147483647;
    return (rng - 1) / 2147483646;
  };

  const current = new Date(start);
  while (current <= end) {
    const dateStr = current.toISOString().split('T')[0];

    // Strategy: slight positive drift + noise
    strategyNAV *= 1 + 0.0008 + (nextRand() - 0.48) * 0.02;
    // Benchmark: smaller drift
    benchmarkNAV *= 1 + 0.0004 + (nextRand() - 0.5) * 0.015;

    data.push({
      date: dateStr,
      strategy: Math.round(strategyNAV * 100) / 100,
      benchmark: Math.round(benchmarkNAV * 100) / 100,
    });

    current.setDate(current.getDate() + 1);
  }

  return data;
}
