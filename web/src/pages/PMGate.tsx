import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Button from '@mui/material/Button';
import Grid from '@mui/material/Grid';
import Chip from '@mui/material/Chip';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';
import DialogActions from '@mui/material/DialogActions';
import TextField from '@mui/material/TextField';
import Checkbox from '@mui/material/Checkbox';
import FormControlLabel from '@mui/material/FormControlLabel';
import WarningIcon from '@mui/icons-material/Warning';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import IconButton from '@mui/material/IconButton';
import Tooltip from '@mui/material/Tooltip';
import Switch from '@mui/material/Switch';
import Stack from '@mui/material/Stack';
import NotificationsActiveIcon from '@mui/icons-material/NotificationsActive';
import NotificationsOffIcon from '@mui/icons-material/NotificationsOff';
import { useState, useEffect, useMemo, useCallback } from 'react';
import { useSnackbar } from 'notistack';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  getGateStatus,
  getGateHistory,
  emergencyStopGate,
  emergencyReopenGate,
  globalEmergencyStop,
  globalEmergencyReopen,
} from '@/lib/api/client';
import type { GateDimension } from '@/types/api';

export function PMGatePage() {
  const queryClient = useQueryClient();
  const { enqueueSnackbar } = useSnackbar();
  const [stopDialogOpen, setStopDialogOpen] = useState(false);
  const [reopenDialogOpen, setReopenDialogOpen] = useState(false);
  const [globalStopDialogOpen, setGlobalStopDialogOpen] = useState(false);
  const [dimension, setDimension] = useState<GateDimension>('signal');
  const [reason, setReason] = useState('');
  const [confirmedStop, setConfirmedStop] = useState(false);
  const [confirmedGlobalStop, setConfirmedGlobalStop] = useState(false);

  // C4: History filters & pagination
  const [historyDimension, setHistoryDimension] = useState<string>('');
  const [historyPage, setHistoryPage] = useState(0);
  const historyLimit = 20;

  // C3: Notification toggle
  const [notificationsEnabled, setNotificationsEnabled] = useState(() => {
    return localStorage.getItem('qlib1_gate_notifications') === 'true';
  });

  const { data: gateStatus, isLoading } = useQuery({
    queryKey: ['gate-status'],
    queryFn: getGateStatus,
    refetchInterval: 3000, // C3: 3s polling
  });

  const historyParams = useMemo(() => ({
    limit: historyLimit,
    dimension: historyDimension || undefined,
  }), [historyLimit, historyDimension]);

  const { data: history } = useQuery({
    queryKey: ['gate-history', historyParams],
    queryFn: () => getGateHistory(historyParams),
  });

  // C3: Browser notification when gate closes
  const prevGateState = useState<string | null>(null);
  useEffect(() => {
    if (!gateStatus || !notificationsEnabled) return;
    const currentState = JSON.stringify(gateStatus.gates);
    if (prevGateState[0] && prevGateState[0] !== currentState) {
      const prev = JSON.parse(prevGateState[0]);
      const curr = gateStatus.gates;
      for (const dim of ['signal', 'train', 'deploy'] as GateDimension[]) {
        if (prev[dim] === 'open' && curr[dim] === 'closed') {
          if (Notification.permission === 'granted') {
            new Notification('Qlib1 Gate Alert', {
              body: `${dim.toUpperCase()} gate has been CLOSED`,
              icon: '/favicon.ico',
            });
          }
        }
      }
    }
    prevGateState[1](currentState);
  }, [gateStatus, notificationsEnabled]);

  const requestNotificationPermission = useCallback(() => {
    if (Notification.permission === 'default') {
      Notification.requestPermission().then((perm) => {
        if (perm === 'granted') {
          setNotificationsEnabled(true);
          localStorage.setItem('qlib1_gate_notifications', 'true');
          enqueueSnackbar('Notifications enabled', { variant: 'success' });
        } else {
          setNotificationsEnabled(false);
          localStorage.removeItem('qlib1_gate_notifications');
        }
      });
    } else {
      const newVal = !notificationsEnabled;
      setNotificationsEnabled(newVal);
      localStorage.setItem('qlib1_gate_notifications', String(newVal));
      enqueueSnackbar(newVal ? 'Notifications enabled' : 'Notifications disabled', {
        variant: newVal ? 'success' : 'info',
      });
    }
  }, [notificationsEnabled, enqueueSnackbar]);

  // C6: Timeline data (last 7 days)
  const timelineData = useMemo(() => {
    if (!history?.history) return [];
    const sevenDaysAgo = new Date();
    sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);
    return history.history
      .filter((e) => new Date(e.timestamp) >= sevenDaysAgo)
      .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
  }, [history]);

  // C4: Relative time helper
  const relativeTime = (ts: string) => {
    const diff = Date.now() - new Date(ts).getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return 'just now';
    if (mins < 60) return `${mins}m ago`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
  };

  const stopMutation = useMutation({
    mutationFn: () => emergencyStopGate({ dimension, reason }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['gate-status'] });
      queryClient.invalidateQueries({ queryKey: ['gate-history'] });
      setStopDialogOpen(false);
      setReason('');
      setConfirmedStop(false);
      enqueueSnackbar(`Emergency stop applied to ${dimension}`, { variant: 'success' });
    },
    onError: () => {
      enqueueSnackbar(`Failed to stop ${dimension}`, { variant: 'error' });
    },
  });

  const reopenMutation = useMutation({
    mutationFn: () => emergencyReopenGate({ dimension, reason }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['gate-status'] });
      queryClient.invalidateQueries({ queryKey: ['gate-history'] });
      setReopenDialogOpen(false);
      setReason('');
      enqueueSnackbar(`${dimension} gate reopened`, { variant: 'success' });
    },
    onError: () => {
      enqueueSnackbar(`Failed to reopen ${dimension}`, { variant: 'error' });
    },
  });

  const globalStopMutation = useMutation({
    mutationFn: () => globalEmergencyStop({ reason }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['gate-status'] });
      queryClient.invalidateQueries({ queryKey: ['gate-history'] });
      setGlobalStopDialogOpen(false);
      setReason('');
      setConfirmedGlobalStop(false);
      enqueueSnackbar('Global emergency stop applied to all dimensions', { variant: 'success' });
    },
    onError: () => {
      enqueueSnackbar('Failed to apply global stop', { variant: 'error' });
    },
  });

  const isGateClosed = gateStatus?.is_any_closed ?? false;

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 600 }}>
          PM Gate Control
        </Typography>
        <Tooltip title={notificationsEnabled ? 'Notifications ON' : 'Notifications OFF'}>
          <IconButton
            onClick={requestNotificationPermission}
            color={notificationsEnabled ? 'warning' : 'default'}
          >
            {notificationsEnabled ? <NotificationsActiveIcon /> : <NotificationsOffIcon />}
          </IconButton>
        </Tooltip>
      </Box>

      <Grid container spacing={3} sx={{ mb: 3 }}>
        {/* Global Emergency Stop */}
        <Grid item xs={12}>
          <Card sx={{ border: '2px solid', borderColor: 'error.main' }}>
            <CardContent sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <WarningIcon color="error" sx={{ fontSize: 36 }} />
                <Box>
                  <Typography variant="h6" sx={{ fontWeight: 700, color: 'error.main' }}>
                    Global Emergency Stop
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Immediately halts ALL gate dimensions (signal, train, deploy)
                  </Typography>
                </Box>
              </Box>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Button
                  variant="contained"
                  color="error"
                  onClick={() => {
                    setDimension('signal');
                    setGlobalStopDialogOpen(true);
                  }}
                  disabled={isGateClosed && gateStatus &&
                    (['signal', 'train', 'deploy'] as GateDimension[]).every((d) => gateStatus.gates[d] === 'closed')}
                >
                  Global Stop All
                </Button>
                <Button
                  variant="contained"
                  color="success"
                  onClick={() => {
                    setDimension('signal');
                    setReopenDialogOpen(true);
                  }}
                  disabled={!isGateClosed}
                >
                  Reopen All
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {gateStatus &&
          (['signal', 'train', 'deploy'] as GateDimension[]).map((dim) => (
            <Grid item xs={12} md={4} key={dim}>
              <Card>
                <CardContent sx={{ textAlign: 'center' }}>
                  <Typography variant="h6" sx={{ textTransform: 'capitalize', mb: 1 }}>
                    {dim}
                  </Typography>
                  <Chip
                    label={gateStatus.gates[dim].toUpperCase()}
                    color={gateStatus.gates[dim] === 'open' ? 'success' : 'error'}
                    sx={{ fontSize: '1.1rem', px: 2, py: 2.5 }}
                  />
                  <Box sx={{ mt: 2, display: 'flex', gap: 1, justifyContent: 'center' }}>
                    <Button
                      variant="outlined"
                      color="error"
                      size="small"
                      disabled={gateStatus.gates[dim] === 'closed'}
                      onClick={() => {
                        setDimension(dim);
                        setStopDialogOpen(true);
                      }}
                    >
                      Stop
                    </Button>
                    <Button
                      variant="outlined"
                      color="success"
                      size="small"
                      disabled={gateStatus.gates[dim] === 'open'}
                      onClick={() => {
                        setDimension(dim);
                        setReopenDialogOpen(true);
                      }}
                    >
                      Reopen
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
      </Grid>

      {/* C6: 7-Day Gate Timeline */}
      {timelineData.length > 0 && (
        <>
          <Typography variant="h5" sx={{ mb: 2 }}>
            Gate Timeline (Last 7 Days)
          </Typography>
          <Paper sx={{ p: 3, mb: 3, overflow: 'auto' }}>
            <Box sx={{ position: 'relative', minWidth: timelineData.length * 28 + 40 }}>
              {/* Timeline bar */}
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                {timelineData.map((entry, i) => (
                  <Tooltip
                    key={entry.action_id}
                    title={
                      <Box>
                        <Typography variant="caption">
                          {entry.dimension} — {entry.action}
                        </Typography>
                        <br />
                        <Typography variant="caption">
                          {new Date(entry.timestamp).toLocaleString()}
                        </Typography>
                        <br />
                        <Typography variant="caption">{entry.reason}</Typography>
                      </Box>
                    }
                  >
                    <Box
                      sx={{
                        width: 20,
                        height: 20,
                        borderRadius: '50%',
                        bgcolor: entry.action === 'emergency_stop' ? 'error.main' : 'success.main',
                        border: '2px solid',
                        borderColor: 'background.paper',
                        ml: i === 0 ? 0 : '8px',
                        flexShrink: 0,
                        cursor: 'pointer',
                      }}
                    />
                  </Tooltip>
                ))}
              </Box>
              {/* Timeline labels */}
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="caption" color="text.secondary">
                  7 days ago
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Now
                </Typography>
              </Box>
            </Box>
          </Paper>
        </>
      )}

      {/* C4: History with filters */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
        <Typography variant="h5">
          Action History
          {history && (
            <Chip label={`${history.total} actions`} size="small" sx={{ ml: 1 }} />
          )}
        </Typography>
        <FormControl size="small" sx={{ minWidth: 150 }}>
          <InputLabel>Dimension</InputLabel>
          <Select
            value={historyDimension}
            label="Dimension"
            onChange={(e) => { setHistoryDimension(e.target.value); setHistoryPage(0); }}
          >
            <MenuItem value="">All</MenuItem>
            <MenuItem value="signal">Signal</MenuItem>
            <MenuItem value="train">Train</MenuItem>
            <MenuItem value="deploy">Deploy</MenuItem>
          </Select>
        </FormControl>
      </Box>
      <TableContainer component={Paper}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Action ID</TableCell>
              <TableCell>Dimension</TableCell>
              <TableCell>Action</TableCell>
              <TableCell>Triggered By</TableCell>
              <TableCell>Reason</TableCell>
              <TableCell>Timestamp</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {history?.history.map((entry) => (
              <TableRow key={entry.action_id}>
                <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>
                  {entry.action_id.slice(0, 8)}...
                </TableCell>
                <TableCell>
                  <Chip label={entry.dimension} size="small" variant="outlined" />
                </TableCell>
                <TableCell>
                  <Chip
                    label={entry.action}
                    color={entry.action === 'emergency_stop' ? 'error' : 'success'}
                    size="small"
                  />
                </TableCell>
                <TableCell>{entry.triggered_by}</TableCell>
                <TableCell sx={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {entry.reason}
                </TableCell>
                <TableCell>
                  <Tooltip title={new Date(entry.timestamp).toLocaleString()}>
                    <Typography variant="body2">{relativeTime(entry.timestamp)}</Typography>
                  </Tooltip>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Stop Dialog */}
      <Dialog open={stopDialogOpen} onClose={() => { setStopDialogOpen(false); setConfirmedStop(false); }}>
        <DialogTitle sx={{ color: 'error.main' }}>Emergency Stop — {dimension}</DialogTitle>
        <DialogContent>
          <DialogContentText sx={{ mb: 2 }}>
            This action will immediately halt all activities in the <strong>{dimension}</strong> dimension.
            Trading signals, model training, or deployments in this dimension will be interrupted.
          </DialogContentText>
          <TextField
            fullWidth
            label="Reason"
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            multiline
            rows={3}
            sx={{ mb: 2, minWidth: 400 }}
          />
          <FormControlLabel
            control={
              <Checkbox
                checked={confirmedStop}
                onChange={(e) => setConfirmedStop(e.target.checked)}
                color="error"
              />
            }
            label={`I confirm this will halt all ${dimension} activities`}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => { setStopDialogOpen(false); setConfirmedStop(false); }}>Cancel</Button>
          <Button
            color="error"
            variant="contained"
            onClick={() => stopMutation.mutate()}
            disabled={!reason.trim() || !confirmedStop}
          >
            Confirm Stop
          </Button>
        </DialogActions>
      </Dialog>

      {/* Reopen Dialog */}
      <Dialog open={reopenDialogOpen} onClose={() => setReopenDialogOpen(false)}>
        <DialogTitle>Emergency Reopen — {dimension}</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            label="Reason"
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            multiline
            rows={3}
            sx={{ mt: 1, minWidth: 400 }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setReopenDialogOpen(false)}>Cancel</Button>
          <Button
            color="success"
            variant="contained"
            onClick={() => reopenMutation.mutate()}
            disabled={!reason.trim()}
          >
            Confirm Reopen
          </Button>
        </DialogActions>
      </Dialog>

      {/* Global Stop Dialog */}
      <Dialog open={globalStopDialogOpen} onClose={() => { setGlobalStopDialogOpen(false); setConfirmedGlobalStop(false); }}>
        <DialogTitle sx={{ color: 'error.main' }}>Global Emergency Stop</DialogTitle>
        <DialogContent>
          <DialogContentText sx={{ mb: 2 }}>
            <strong>CRITICAL:</strong> This will immediately halt ALL gate dimensions
            (signal, train, deploy). All trading and model operations will be suspended.
          </DialogContentText>
          <TextField
            fullWidth
            label="Reason"
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            multiline
            rows={3}
            sx={{ mb: 2, minWidth: 400 }}
          />
          <FormControlLabel
            control={
              <Checkbox
                checked={confirmedGlobalStop}
                onChange={(e) => setConfirmedGlobalStop(e.target.checked)}
                color="error"
              />
            }
            label="I confirm this will halt ALL dimensions (signal, train, deploy)"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => { setGlobalStopDialogOpen(false); setConfirmedGlobalStop(false); }}>Cancel</Button>
          <Button
            color="error"
            variant="contained"
            onClick={() => globalStopMutation.mutate()}
            disabled={!reason.trim() || !confirmedGlobalStop}
          >
            Confirm Global Stop
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
