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
import DialogActions from '@mui/material/DialogActions';
import TextField from '@mui/material/TextField';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
import { useState } from 'react';
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
  const [stopDialogOpen, setStopDialogOpen] = useState(false);
  const [reopenDialogOpen, setReopenDialogOpen] = useState(false);
  const [dimension, setDimension] = useState<GateDimension>('signal');
  const [reason, setReason] = useState('');

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
    },
  });

  const reopenMutation = useMutation({
    mutationFn: () => emergencyReopenGate({ dimension, reason }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['gate-status'] });
      queryClient.invalidateQueries({ queryKey: ['gate-history'] });
      setReopenDialogOpen(false);
      setReason('');
    },
  });

  const isGateClosed = gateStatus?.is_any_closed ?? false;

  return (
    <Box>
      <Typography variant="h4" sx={{ mb: 3, fontWeight: 600 }}>
        PM Gate Control
      </Typography>

      <Grid container spacing={3} sx={{ mb: 3 }}>
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
      <Dialog open={stopDialogOpen} onClose={() => setStopDialogOpen(false)}>
        <DialogTitle>Emergency Stop - {dimension}</DialogTitle>
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
          <Button onClick={() => setStopDialogOpen(false)}>Cancel</Button>
          <Button
            color="error"
            variant="contained"
            onClick={() => stopMutation.mutate()}
            disabled={!reason.trim()}
          >
            Confirm Stop
          </Button>
        </DialogActions>
      </Dialog>

      {/* Reopen Dialog */}
      <Dialog open={reopenDialogOpen} onClose={() => setReopenDialogOpen(false)}>
        <DialogTitle>Emergency Reopen - {dimension}</DialogTitle>
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
    </Box>
  );
}
