import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Chip from '@mui/material/Chip';

interface StatusCardProps {
  title: string;
  status: 'healthy' | 'unhealthy' | 'loading';
  subtitle?: string;
}

export function StatusCard({ title, status, subtitle }: StatusCardProps) {
  const getStatusColor = () => {
    switch (status) {
      case 'healthy':
        return 'success';
      case 'unhealthy':
        return 'error';
      default:
        return 'default';
    }
  };

  const getStatusLabel = () => {
    switch (status) {
      case 'healthy':
        return 'Healthy';
      case 'unhealthy':
        return 'Unhealthy';
      default:
        return 'Loading...';
    }
  };

  return (
    <Box
      sx={{
        p: 2,
        borderRadius: 2,
        bgcolor: 'background.paper',
        border: '1px solid',
        borderColor: 'divider',
      }}
    >
      <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
        {title}
      </Typography>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <Chip
          label={getStatusLabel()}
          color={getStatusColor()}
          size="small"
          sx={{ fontWeight: 600 }}
        />
        {subtitle && (
          <Typography variant="caption" color="text.secondary">
            {subtitle}
          </Typography>
        )}
      </Box>
    </Box>
  );
}