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

interface FactorTimePoint {
  date: string;
  [factorName: string]: string | number;
}

interface FactorTimeSeriesProps {
  data: FactorTimePoint[];
  factors: string[];
  selectedFactor?: string;
}

export function FactorTimeSeries({ data, factors, selectedFactor }: FactorTimeSeriesProps) {
  const theme = useUIStore((s) => s.theme);
  const isDark = theme === 'dark' || (theme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);

  const colors = ['#0ea5e9', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];
  const grid = isDark ? '#334155' : '#e2e8f0';
  const text = isDark ? '#94a3b8' : '#64748b';

  // Only show selected factor or first 3
  const displayFactors = selectedFactor ? [selectedFactor] : factors.slice(0, 3);

  return (
    <Box>
      <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
        Factor Time Series
      </Typography>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data} margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={grid} />
          <XAxis dataKey="date" stroke={text} tick={{ fontSize: 11 }} tickLine={false} />
          <YAxis stroke={text} tick={{ fontSize: 11 }} tickLine={false} />
          <Tooltip
            contentStyle={{
              backgroundColor: isDark ? '#1e293b' : '#fff',
              border: `1px solid ${grid}`,
              borderRadius: 8,
              fontSize: 12,
            }}
          />
          <Legend />
          {displayFactors.map((factor, idx) => (
            <Line
              key={factor}
              type="monotone"
              dataKey={factor}
              stroke={colors[idx % colors.length]}
              strokeWidth={1.5}
              dot={false}
              name={factor}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </Box>
  );
}

/** Generate mock factor time series data */
export function generateMockFactorTimeSeries(
  factors: string[],
  dates: string[]
): FactorTimePoint[] {
  let rng = 42;
  const nextRand = () => {
    rng = (rng * 16807 + 0) % 2147483647;
    return (rng - 1) / 2147483646;
  };

  return dates.map((date) => {
    const point: FactorTimePoint = { date };
    for (const factor of factors) {
      point[factor] = Math.round((nextRand() * 4 - 2) * 1000) / 1000;
    }
    return point;
  });
}
