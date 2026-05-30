import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';

interface QuickStatsCardProps {
  title: string;
  value: string;
  subtitle?: string;
  status?: 'success' | 'warning' | 'error' | 'info';
}

export function QuickStatsCard({ title, value, subtitle, status = 'info' }: QuickStatsCardProps) {
  const statusColors = {
    success: '#22c55e',
    warning: '#f59e0b',
    error: '#ef4444',
    info: '#0ea5e9',
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
          {title}
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 1 }}>
          <Typography variant="h4" sx={{ fontWeight: 700, color: statusColors[status] }}>
            {value}
          </Typography>
        </Box>
        {subtitle && (
          <Typography variant="caption" color="text.secondary">
            {subtitle}
          </Typography>
        )}
      </CardContent>
    </Card>
  );
}