import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Tooltip from '@mui/material/Tooltip';

interface MetricsCardProps {
  label: string;
  value: string;
  subtitle?: string;
  threshold?: 'good' | 'warn' | 'bad' | 'neutral';
  tooltip?: string;
  format?: 'number' | 'percentage' | 'ratio';
}

const thresholdColors = {
  good: { bg: '#dcfce7', text: '#166534', darkBg: '#064e3b', darkText: '#86efac' },
  warn: { bg: '#fef9c3', text: '#854d0e', darkBg: '#422006', darkText: '#fde047' },
  bad: { bg: '#fee2e2', text: '#991b1b', darkBg: '#450a0a', darkText: '#fca5a5' },
  neutral: { bg: '#f1f5f9', text: '#334155', darkBg: '#1e293b', darkText: '#94a3b8' },
};

export function MetricsCard({
  label,
  value,
  subtitle,
  threshold = 'neutral',
  tooltip,
}: MetricsCardProps) {
  const isDark = typeof window !== 'undefined' &&
    (document.documentElement.getAttribute('data-mui-color-scheme') === 'dark');

  const colors = thresholdColors[threshold];

  const card = (
    <Card
      sx={{
        bgcolor: isDark ? colors.darkBg : colors.bg,
        border: 'none',
        height: '100%',
      }}
    >
      <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
        <Typography
          variant="caption"
          sx={{
            color: isDark ? colors.darkText : colors.text,
            textTransform: 'uppercase',
            fontWeight: 600,
            letterSpacing: 0.5,
          }}
        >
          {label}
        </Typography>
        <Typography
          variant="h5"
          sx={{
            fontWeight: 700,
            color: isDark ? colors.darkText : colors.text,
            mt: 0.5,
          }}
        >
          {value}
        </Typography>
        {subtitle && (
          <Typography
            variant="caption"
            sx={{
              color: isDark ? colors.darkText : colors.text,
              opacity: 0.7,
            }}
          >
            {subtitle}
          </Typography>
        )}
      </CardContent>
    </Card>
  );

  if (tooltip) {
    return <Tooltip title={tooltip}>{card}</Tooltip>;
  }

  return card;
}
