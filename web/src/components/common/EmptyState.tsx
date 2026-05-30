import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import CircularProgress from '@mui/material/CircularProgress';
import type { SvgIconComponent } from '@mui/icons-material';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';

interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
  loading?: boolean;
  loadingText?: string;
}

export function EmptyState({
  icon,
  title,
  description,
  action,
  loading = false,
  loadingText,
}: EmptyStateProps) {
  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: 200,
        gap: 2,
        py: 6,
        px: 3,
        textAlign: 'center',
      }}
    >
      {loading ? (
        <CircularProgress size={40} />
      ) : (
        icon || <InfoOutlinedIcon sx={{ fontSize: 48, color: 'text.disabled' }} />
      )}
      <Typography variant="h6" sx={{ fontWeight: 600, color: loading ? 'text.secondary' : 'text.primary' }}>
        {loading ? (loadingText || title) : title}
      </Typography>
      {description && !loading && (
        <Typography variant="body2" color="text.secondary" sx={{ maxWidth: 400 }}>
          {description}
        </Typography>
      )}
      {action && !loading && (
        <Button variant="outlined" onClick={action.onClick} sx={{ mt: 1 }}>
          {action.label}
        </Button>
      )}
    </Box>
  );
}
