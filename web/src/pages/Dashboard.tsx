import { AppShell } from '@/components/layout/AppShell';
import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Skeleton from '@mui/material/Skeleton';
import Alert from '@mui/material/Alert';
import Button from '@mui/material/Button';
import ToggleButton from '@mui/material/ToggleButton';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';
import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { getHealth, getSystemMetrics, getGateStatus } from '@/lib/api/client';
import { StatusCard } from '@/components/common/StatusCard';
import { QuickStatsCard } from '@/components/dashboard/QuickStatsCard';
import { EmptyState } from '@/components/common/EmptyState';

type PollInterval = 10_000 | 30_000 | 60_000 | 0;

export function DashboardPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [pollInterval, setPollInterval] = useState<PollInterval>(30_000);
  const [healthTimedOut, setHealthTimedOut] = useState(false);
  const healthTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const { data: health, isLoading: healthLoading, isError: healthError } = useQuery({
    queryKey: ['health'],
    queryFn: getHealth,
    refetchInterval: pollInterval || false,
  });

  // Timeout detection: show EmptyState if health check exceeds 10s
  useEffect(() => {
    if (healthLoading && !health) {
      healthTimerRef.current = setTimeout(() => setHealthTimedOut(true), 10_000);
    } else {
      if (healthTimerRef.current) clearTimeout(healthTimerRef.current);
      setHealthTimedOut(false);
    }
    return () => {
      if (healthTimerRef.current) clearTimeout(healthTimerRef.current);
    };
  }, [healthLoading, health]);

  const { data: metrics } = useQuery({
    queryKey: ['metrics'],
    queryFn: getSystemMetrics,
    enabled: health?.status === 'healthy',
    refetchInterval: pollInterval || false,
  });

  const { data: gateStatus } = useQuery({
    queryKey: ['gate-status'],
    queryFn: getGateStatus,
    refetchInterval: pollInterval || false,
  });

  const isUnhealthy = health && health.status !== 'healthy';
  const isAnyGateClosed = gateStatus?.is_any_closed ?? false;
  const showAlert = isUnhealthy || isAnyGateClosed;

  return (
    <Box>
      {/* Alert banner */}
      {showAlert && (
        <Alert
          severity="error"
          sx={{ mb: 3 }}
          action={
            isAnyGateClosed ? (
              <Button color="inherit" size="small" onClick={() => navigate('/gate')}>
                Manage Gate
              </Button>
            ) : undefined
          }
        >
          {isUnhealthy && 'System health is degraded. '}
          {isAnyGateClosed && 'One or more PM Gates are CLOSED — trading may be halted.'}
        </Alert>
      )}

      {/* Polling interval controls */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 1 }}>
        <Typography variant="h4" sx={{ fontWeight: 600 }}>
          Dashboard
        </Typography>
        <ToggleButtonGroup
          value={pollInterval}
          exclusive
          onChange={(_, val) => val !== null && setPollInterval(val)}
          size="small"
        >
          <ToggleButton value={10_000}>10s</ToggleButton>
          <ToggleButton value={30_000}>30s</ToggleButton>
          <ToggleButton value={60_000}>60s</ToggleButton>
          <ToggleButton value={0}>Manual</ToggleButton>
        </ToggleButtonGroup>
      </Box>

      {/* Timeout / Error state */}
      {healthTimedOut && !health && (
        <EmptyState
          title="Connection Timeout"
          description="Unable to reach the server after 10 seconds. The backend service may be starting up or unavailable."
          loading={healthLoading}
          loadingText="Reconnecting..."
          action={{
            label: 'Retry Connection',
            onClick: () => {
              setHealthTimedOut(false);
              queryClient.invalidateQueries({ queryKey: ['health'] });
            },
          }}
        />
      )}
      {healthError && !health && !healthTimedOut && (
        <EmptyState
          title="Service Unavailable"
          description="The backend service returned an error. Check that the API server is running."
          action={{
            label: 'Retry',
            onClick: () => queryClient.invalidateQueries({ queryKey: ['health'] }),
          }}
        />
      )}

      {/* Main content — always render with skeletons when loading */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <StatusCard
            title="System Health"
            status={healthLoading ? 'loading' : health?.status === 'healthy' ? 'healthy' : 'unhealthy'}
            subtitle={healthLoading ? 'Checking...' : `Version ${health?.version}`}
          />
        </Grid>

        <Grid item xs={12} md={4}>
          {metrics ? (
            <QuickStatsCard
              title="Cache Hit Rate"
              value={`${(metrics.cache_hit_rate * 100).toFixed(1)}%`}
              subtitle={`${metrics.cache_size} cached items`}
              status={metrics.cache_hit_rate > 0.8 ? 'success' : metrics.cache_hit_rate > 0.6 ? 'warning' : 'error'}
            />
          ) : (
            <Card>
              <CardContent>
                <Skeleton variant="text" width="40%" />
                <Skeleton variant="text" width="60%" height={48} />
              </CardContent>
            </Card>
          )}
        </Grid>

        <Grid item xs={12} md={4}>
          {metrics ? (
            <QuickStatsCard
              title="Active Models"
              value={metrics.active_models.toString()}
              subtitle={`${metrics.requests_total} total requests`}
              status="info"
            />
          ) : (
            <Card>
              <CardContent>
                <Skeleton variant="text" width="40%" />
                <Skeleton variant="text" width="60%" height={48} />
              </CardContent>
            </Card>
          )}
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>
                PM Gate Status
              </Typography>
              {gateStatus ? (
                <Box sx={{ display: 'flex', gap: 2 }}>
                  {(['signal', 'train', 'deploy'] as const).map((dim) => (
                    <Box
                      key={dim}
                      sx={{
                        flex: 1,
                        p: 2,
                        borderRadius: 1,
                        bgcolor: gateStatus.gates[dim] === 'open' ? 'success.light' : 'error.light',
                        color: gateStatus.gates[dim] === 'open' ? 'success.dark' : 'error.dark',
                        textAlign: 'center',
                      }}
                    >
                      <Typography variant="caption" sx={{ textTransform: 'uppercase' }}>
                        {dim}
                      </Typography>
                      <Typography variant="h5" sx={{ fontWeight: 700 }}>
                        {gateStatus.gates[dim].toUpperCase()}
                      </Typography>
                    </Box>
                  ))}
                </Box>
              ) : (
                <Skeleton variant="rectangular" height={80} />
              )}
              <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
                <Button
                  size="small"
                  variant="outlined"
                  onClick={() => navigate('/gate')}
                  sx={{ fontSize: 12 }}
                >
                  Manage Gate
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>
                Performance
              </Typography>
              {metrics ? (
                <Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">Avg Latency</Typography>
                    <Typography
                      variant="body2"
                      sx={{ color: metrics.avg_latency_ms > 200 ? 'error.main' : 'success.main' }}
                    >
                      {metrics.avg_latency_ms.toFixed(1)}ms
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2">Uptime</Typography>
                    <Typography variant="body2">
                      {Math.floor(metrics.uptime_seconds / 3600)}h {Math.floor((metrics.uptime_seconds % 3600) / 60)}m
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2">Cache Evictions</Typography>
                    <Typography variant="body2">{metrics.cache_evictions}</Typography>
                  </Box>
                </Box>
              ) : (
                <Skeleton variant="text" />
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}
