'use client';

import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import { AppShell } from '@/components/layout/AppShell';

export default function SettingsPage() {
  return (
    <AppShell>
      <Box>
        <Typography variant="h4" sx={{ mb: 3, fontWeight: 600 }}>
          Settings
        </Typography>

        <Card sx={{ maxWidth: 600 }}>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 2 }}>
              API Configuration
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <TextField
                label="API Base URL"
                defaultValue={process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000'}
                size="small"
                helperText="Backend API endpoint"
              />
              <Button variant="outlined">Save Changes</Button>
            </Box>
          </CardContent>
        </Card>
      </Box>
    </AppShell>
  );
}