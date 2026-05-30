import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import { useUIStore } from '@/store/uiStore';

interface HistogramBin {
  bin: string;
  count: number;
}

interface FactorHistogramProps {
  data: HistogramBin[];
  factorName: string;
}

export function FactorHistogram({ data, factorName }: FactorHistogramProps) {
  const theme = useUIStore((s) => s.theme);
  const isDark = theme === 'dark' || (theme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);

  const grid = isDark ? '#334155' : '#e2e8f0';
  const text = isDark ? '#94a3b8' : '#64748b';

  return (
    <Box>
      <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
        {factorName} Distribution
      </Typography>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data} margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={grid} />
          <XAxis dataKey="bin" stroke={text} tick={{ fontSize: 10 }} tickLine={false} />
          <YAxis stroke={text} tick={{ fontSize: 11 }} tickLine={false} />
          <Tooltip
            contentStyle={{
              backgroundColor: isDark ? '#1e293b' : '#fff',
              border: `1px solid ${grid}`,
              borderRadius: 8,
              fontSize: 12,
            }}
          />
          <Bar dataKey="count" fill="#0ea5e9" radius={[4, 4, 0, 0]}>
            {data.map((entry, index) => (
              <Cell key={index} fill={isDark ? '#38bdf8' : '#0ea5e9'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </Box>
  );
}

/** Generate mock histogram bins */
export function generateMockHistogram(factorName: string, numBins = 10): HistogramBin[] {
  let rng = factorName.length * 7;
  const nextRand = () => {
    rng = (rng * 16807 + 0) % 2147483647;
    return (rng - 1) / 2147483646;
  };

  const bins: HistogramBin[] = [];
  const step = 0.5;
  const start = -2.5;

  for (let i = 0; i < numBins; i++) {
    const low = start + i * step;
    const high = low + step;
    bins.push({
      bin: `${low.toFixed(1)}~${high.toFixed(1)}`,
      count: Math.floor(nextRand() * 50 + 5),
    });
  }

  return bins;
}
