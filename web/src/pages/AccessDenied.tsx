import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import LockIcon from '@mui/icons-material/Lock';
import { useNavigate, useLocation } from 'react-router-dom';

export function AccessDeniedPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const state = location.state as { required?: string; role?: string } | null;

  return (
    <Box
      sx={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '60vh',
      }}
    >
      <Card sx={{ maxWidth: 480, textAlign: 'center' }}>
        <CardContent sx={{ py: 5 }}>
          <LockIcon sx={{ fontSize: 64, color: 'error.main', mb: 2 }} />
          <Typography variant="h4" sx={{ mb: 1, fontWeight: 600 }}>
            Access Denied
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
            You do not have the required permissions to access this page.
          </Typography>
          {state?.required && (
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              Required permission: <strong>{state.required}</strong>
            </Typography>
          )}
          {state?.role && (
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Your role: <strong>{state.role}</strong>
            </Typography>
          )}
          <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
            <Button variant="outlined" onClick={() => navigate(-1)}>
              Go Back
            </Button>
            <Button variant="contained" onClick={() => navigate('/')}>
              Dashboard
            </Button>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
}
