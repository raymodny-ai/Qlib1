import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import { useUIStore } from '@/store/uiStore';

interface DrawdownPoint {
  date: string;
  drawdown: number;
}

interface DrawdownChartProps {
  data: DrawdownPoint[];
}

export function DrawdownChart({ data }: DrawdownChartProps) {
  const theme = useUIStore((s) => s.theme);
  const isDark = theme === 'dark' || (theme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);

  const colors = {
    fill: '#fecaca',
    stroke: '#ef4444',
    grid: isDark ? '#334155' : '#e2e8f0',
    text: isDark ? '#94a3b8' : '#64748b',
  };

  const maxDD = Math.min(...data.map((d) => d.drawdown));
  const maxDDDate = data.find((d) => d.drawdown === maxDD)?.date;

  return (
    <Box>
      <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
        Drawdown
        {maxDDDate && (
          <Typography component="span" variant="caption" color="error.main" sx={{ ml: 1 }}>
            Max: {(maxDD * 100).toFixed(2)}%
          </Typography>
        )}
      </Typography>
      <ResponsiveContainer width="100%" height={200}>
        <AreaChart data={data} margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
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
            tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
            domain={[Math.min(-0.5, maxDD * 1.1), 0]}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: isDark ? '#1e293b' : '#fff',
              border: `1px solid ${colors.grid}`,
              borderRadius: 8,
              fontSize: 12,
            }}
            formatter={(value: number) => [`${(value * 100).toFixed(2)}%`, 'Drawdown']}
          />
          <ReferenceLine y={0} stroke={colors.grid} />
          <Area
            type="monotone"
            dataKey="drawdown"
            stroke={colors.stroke}
            fill={colors.fill}
            fillOpacity={0.5}
          />
        </AreaChart>
      </ResponsiveContainer>
    </Box>
  );
}

/** Generate mock drawdown data from equity curve */
export function generateMockDrawdown(
  equityData: { date: string; strategy: number }[]
): DrawdownPoint[] {
  const data: DrawdownPoint[] = [];
  let peak = -Infinity;

  for (const point of equityData) {
    if (point.strategy > peak) {
      peak = point.strategy;
    }
    const dd = peak > 0 ? (point.strategy - peak) / peak : 0;
    data.push({
      date: point.date,
      drawdown: Math.round(dd * 10000) / 10000,
    });
  }

  return data;
}
