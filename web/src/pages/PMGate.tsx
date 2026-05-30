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
import { useState } from 'react';
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

  const { data: gateStatus, isLoading } = useQuery({
    queryKey: ['gate-status'],
    queryFn: getGateStatus,
    refetchInterval: 10000,
  });

  const { data: history } = useQuery({
    queryKey: ['gate-history'],
    queryFn: () => getGateHistory({ limit: 50 }),
  });

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
      <Typography variant="h4" sx={{ mb: 3, fontWeight: 600 }}>
        PM Gate Control
      </Typography>

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

      <Typography variant="h5" sx={{ mb: 2 }}>
        Action History
      </Typography>
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
                <TableCell>{entry.action_id}</TableCell>
                <TableCell>{entry.dimension}</TableCell>
                <TableCell>
                  <Chip
                    label={entry.action}
                    color={entry.action === 'emergency_stop' ? 'error' : 'success'}
                    size="small"
                  />
                </TableCell>
                <TableCell>{entry.triggered_by}</TableCell>
                <TableCell>{entry.reason}</TableCell>
                <TableCell>{new Date(entry.timestamp).toLocaleString()}</TableCell>
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
