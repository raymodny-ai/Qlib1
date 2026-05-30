'use client';

import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Chip from '@mui/material/Chip';
import Skeleton from '@mui/material/Skeleton';
import Alert from '@mui/material/Alert';
import { useQuery } from '@tanstack/react-query';
import { getComplianceStatus, generateSOXReport } from '@/lib/api/client';
import { AppShell } from '@/components/layout/AppShell';
import Button from '@mui/material/Button';

export default function CompliancePage() {
  const { data: status, isLoading } = useQuery({
    queryKey: ['compliance-status'],
    queryFn: getComplianceStatus,
  });

  const soxMutation = useQuery({
    queryKey: ['sox-report'],
    queryFn: () => generateSOXReport({ quarter: '2026-Q2' }),
    enabled: false,
  });

  return (
    <AppShell>
      <Box>
        <Typography variant="h4" sx={{ mb: 3, fontWeight: 600 }}>
          Compliance Status
        </Typography>

        {/* Overall Status */}
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
              <Typography variant="h6">Overall Status</Typography>
              {status && (
                <Chip
                  label={status.overall_status.toUpperCase()}
                  color={
                    status.overall_status === 'compliant'
                      ? 'success'
                      : status.overall_status === 'warning'
                      ? 'warning'
                      : 'error'
                  }
                  sx={{ fontWeight: 700 }}
                />
              )}
            </Box>
            <Typography variant="body2" color="text.secondary">
              Period: {status?.period || 'Loading...'}
            </Typography>
            {status?.audit_chain_verified && (
              <Chip label="Audit Chain Verified" color="success" size="small" sx={{ mt: 1 }} />
            )}
          </CardContent>
        </Card>

        {/* Control Points */}
        <Typography variant="h6" sx={{ mb: 2 }}>
          SOX Control Points
        </Typography>
        <Grid container spacing={2} sx={{ mb: 3 }}>
          {isLoading
            ? Array.from({ length: 6 }).map((_, i) => (
                <Grid item xs={12} md={6} key={i}>
                  <Card>
                    <CardContent>
                      <Skeleton width="30%" />
                      <Skeleton width="60%" />
                    </CardContent>
                  </Card>
                </Grid>
              ))
            : status?.controls.map((control) => (
                <Grid item xs={12} md={6} key={control.control_id}>
                  <Card
                    sx={{
                      borderLeft: '4px solid',
                      borderColor:
                        control.status === 'passed'
                          ? 'success.main'
                          : control.status === 'warning'
                          ? 'warning.main'
                          : 'error.main',
                    }}
                  >
                    <CardContent>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Typography variant="subtitle2">{control.control_id}</Typography>
                        <Chip
                          label={control.status.toUpperCase()}
                          color={
                            control.status === 'passed'
                              ? 'success'
                              : control.status === 'warning'
                              ? 'warning'
                              : 'error'
                          }
                          size="small"
                        />
                      </Box>
                      {control.details && (
                        <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                          {control.details}
                        </Typography>
                      )}
                    </CardContent>
                  </Card>
                </Grid>
              ))}
        </Grid>

        {/* Generate SOX Report */}
        <Card>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 2 }}>
              Generate SOX Report
            </Typography>
            <Button variant="outlined" onClick={() => soxMutation.refetch()}>
              Generate 2026-Q2 Report
            </Button>
            {soxMutation.data && (
              <Alert severity="success" sx={{ mt: 2 }}>
                Report generated at {new Date(soxMutation.data.generated_at).toLocaleString()}
              </Alert>
            )}
          </CardContent>
        </Card>
      </Box>
    </AppShell>
  );
}