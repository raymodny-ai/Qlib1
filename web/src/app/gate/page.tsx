'use client';

import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Grid from '@mui/material/Grid';
import Alert from '@mui/material/Alert';
import Chip from '@mui/material/Chip';
import { useQuery } from '@tanstack/react-query';
import { getGateStatus, emergencyStopGate, emergencyReopenGate, getGateHistory } from '@/lib/api/client';
import { useGateStore } from '@/store/gateStore';
import { AppShell } from '@/components/layout/AppShell';
import Button from '@mui/material/Button';
import TextField from '@mui/material/TextField';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import { useState } from 'react';
import type { GateDimension } from '@/types/api';

export default function GatePage() {
  const { status, setStatus, setHistory } = useGateStore();
  const [dialogOpen, setDialogOpen] = useState(false);
  const [actionType, setActionType] = useState<'stop' | 'reopen'>('stop');
  const [selectedDimension, setSelectedDimension] = useState<GateDimension>('signal');
  const [reason, setReason] = useState('');
  const [loading, setLoading] = useState(false);

  useQuery({
    queryKey: ['gate-status'],
    queryFn: async () => {
      const data = await getGateStatus();
      setStatus(data);
      return data;
    },
    refetchInterval: 10000,
  });

  useQuery({
    queryKey: ['gate-history'],
    queryFn: async () => {
      const data = await getGateHistory({ limit: 20 });
      setHistory(data.history);
      return data;
    },
  });

  const handleAction = async () => {
    setLoading(true);
    try {
      if (actionType === 'stop') {
        await emergencyStopGate({ dimension: selectedDimension, reason });
      } else {
        await emergencyReopenGate({ dimension: selectedDimension, reason });
      }
      // Refresh status
      const newStatus = await getGateStatus();
      setStatus(newStatus);
      setDialogOpen(false);
      setReason('');
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const openDialog = (type: 'stop' | 'reopen', dimension: GateDimension) => {
    setActionType(type);
    setSelectedDimension(dimension);
    setDialogOpen(true);
  };

  return (
    <AppShell>
      <Box>
        <Typography variant="h4" sx={{ mb: 3, fontWeight: 600 }}>
          PM Gate Control
        </Typography>

        {/* Global Alert */}
        {status?.is_any_closed && (
          <Alert severity="error" sx={{ mb: 3 }}>
            One or more gates are closed. Trading signals may be blocked.
          </Alert>
        )}

        {/* Gate Status Cards */}
        <Grid container spacing={3} sx={{ mb: 4 }}>
          {(['signal', 'train', 'deploy'] as const).map((dimension) => {
            const isOpen = status?.gates[dimension] === 'open';
            return (
              <Grid item xs={12} md={4} key={dimension}>
                <Card
                  sx={{
                    border: '2px solid',
                    borderColor: isOpen ? 'success.main' : 'error.main',
                  }}
                >
                  <CardContent>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                      <Typography variant="h6" sx={{ textTransform: 'uppercase' }}>
                        {dimension}
                      </Typography>
                      <Chip
                        label={isOpen ? 'OPEN' : 'CLOSED'}
                        color={isOpen ? 'success' : 'error'}
                        sx={{ fontWeight: 700 }}
                      />
                    </Box>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      {dimension === 'signal' && 'Controls trading signal push to downstream'}
                      {dimension === 'train' && 'Controls model training tasks'}
                      {dimension === 'deploy' && 'Controls model deployment to production'}
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 1 }}>
                      <Button
                        variant="contained"
                        color="error"
                        size="small"
                        onClick={() => openDialog('stop', dimension)}
                        disabled={!isOpen}
                      >
                        Emergency Stop
                      </Button>
                      <Button
                        variant="outlined"
                        color="success"
                        size="small"
                        onClick={() => openDialog('reopen', dimension)}
                        disabled={isOpen}
                      >
                        Reopen
                      </Button>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            );
          })}
        </Grid>

        {/* History */}
        <Card>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 2 }}>
              Recent Actions
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {status?.stats.total_actions || 0} total actions recorded
            </Typography>
          </CardContent>
        </Card>

        {/* Action Dialog */}
        <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)}>
          <DialogTitle>
            {actionType === 'stop' ? 'Emergency Stop' : 'Reopen'} - {selectedDimension.toUpperCase()}
          </DialogTitle>
          <DialogContent>
            <TextField
              label="Reason"
              multiline
              rows={3}
              fullWidth
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              sx={{ mt: 2 }}
              required
            />
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setDialogOpen(false)}>Cancel</Button>
            <Button
              onClick={handleAction}
              color={actionType === 'stop' ? 'error' : 'success'}
              disabled={!reason || loading}
            >
              {loading ? 'Processing...' : 'Confirm'}
            </Button>
          </DialogActions>
        </Dialog>
      </Box>
    </AppShell>
  );
}