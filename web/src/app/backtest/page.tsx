'use client';

import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Alert from '@mui/material/Alert';
import CircularProgress from '@mui/material/CircularProgress';
import LinearProgress from '@mui/material/LinearProgress';
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { useMutation } from '@tanstack/react-query';
import { useBacktestStore } from '@/store/backtestStore';
import { submitBacktest, getBacktestStatus } from '@/lib/api/client';
import { AppShell } from '@/components/layout/AppShell';
import { useEffect, useCallback } from 'react';

export default function BacktestPage() {
  const { addTask, updateTaskStatus, getCurrentTask, isSubmitting, setSubmitting } = useBacktestStore();

  const mutation = useMutation({
    mutationFn: submitBacktest,
    onSuccess: (data) => {
      addTask(data.task_id, mutation.variables!, data);
      // Poll for status updates
      startPolling(data.task_id);
    },
  });

  const handleSubmit = useCallback(() => {
    setSubmitting(true);
    mutation.mutate({
      strategy_type: 'topk_dropout',
      model_name: 'lightgbm',
      start_date: '2024-01-01',
      end_date: '2025-12-31',
      initial_capital: 1000000,
      top_k: 30,
      rebalance_freq: 5,
      commission_rate: 0.001,
    });
  }, []);

  const startPolling = useCallback((taskId: string) => {
    const interval = setInterval(async () => {
      try {
        const status = await getBacktestStatus(taskId);
        updateTaskStatus(taskId, status);
        if (status.status === 'completed' || status.status === 'failed') {
          clearInterval(interval);
          setSubmitting(false);
        }
      } catch (e) {
        clearInterval(interval);
      }
    }, 2000);
  }, []);

  const currentTask = getCurrentTask();
  const result = currentTask?.status.result;

  return (
    <AppShell>
      <Box>
        <Typography variant="h4" sx={{ mb: 3, fontWeight: 600 }}>
          Backtest
        </Typography>

        <Grid container spacing={3}>
          {/* Configuration Form */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 2 }}>
                  Configuration
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <FormControl size="small">
                    <InputLabel>Strategy</InputLabel>
                    <Select label="Strategy" defaultValue="topk_dropout">
                      <MenuItem value="topk_dropout">Top-K Dropout</MenuItem>
                      <MenuItem value="equal_weight">Equal Weight</MenuItem>
                      <MenuItem value="score_weight">Score Weight</MenuItem>
                    </Select>
                  </FormControl>
                  <TextField
                    label="Model"
                    defaultValue="lightgbm"
                    size="small"
                  />
                  <Box sx={{ display: 'flex', gap: 2 }}>
                    <TextField
                      label="Start Date"
                      type="date"
                      size="small"
                      defaultValue="2024-01-01"
                      InputLabelProps={{ shrink: true }}
                    />
                    <TextField
                      label="End Date"
                      type="date"
                      size="small"
                      defaultValue="2025-12-31"
                      InputLabelProps={{ shrink: true }}
                    />
                  </Box>
                  <TextField
                    label="Initial Capital"
                    type="number"
                    defaultValue={1000000}
                    size="small"
                    InputProps={{ startAdornment: '$' }}
                  />
                  <Box sx={{ display: 'flex', gap: 2 }}>
                    <TextField
                      label="Top-K"
                      type="number"
                      defaultValue={30}
                      size="small"
                    />
                    <TextField
                      label="Rebalance Freq"
                      type="number"
                      defaultValue={5}
                      size="small"
                    />
                  </Box>

                  <Button
                    variant="contained"
                    onClick={handleSubmit}
                    disabled={isSubmitting}
                    sx={{ mt: 2 }}
                  >
                    {isSubmitting ? <CircularProgress size={24} /> : 'Run Backtest'}
                  </Button>

                  {mutation.isError && (
                    <Alert severity="error">{mutation.error.message}</Alert>
                  )}
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Results */}
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 2 }}>
                  Results
                </Typography>
                {currentTask && (
                  <Box>
                    <Typography variant="body2" sx={{ mb: 2 }}>
                      Task ID: {currentTask.taskId} | Status: {currentTask.status.status}
                    </Typography>
                    {currentTask.status.status === 'running' && (
                      <LinearProgress />
                    )}
                    {currentTask.status.status === 'failed' && (
                      <Alert severity="error">{currentTask.status.error}</Alert>
                    )}
                  </Box>
                )}
                {!currentTask && (
                  <Typography variant="body2" color="text.secondary">
                    Run a backtest to see results
                  </Typography>
                )}
              </CardContent>
            </Card>

            {/* KPI Cards */}
            {result && (
              <Grid container spacing={2} sx={{ mt: 1 }}>
                <Grid item xs={6}>
                  <Card sx={{ bgcolor: 'success.light' }}>
                    <CardContent>
                      <Typography variant="caption">Total Return</Typography>
                      <Typography variant="h5">{(result.total_return * 100).toFixed(2)}%</Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={6}>
                  <Card sx={{ bgcolor: 'primary.light' }}>
                    <CardContent>
                      <Typography variant="caption">Sharpe Ratio</Typography>
                      <Typography variant="h5">{result.sharpe_ratio.toFixed(2)}</Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={6}>
                  <Card sx={{ bgcolor: result.max_drawdown < -0.15 ? 'error.light' : 'warning.light' }}>
                    <CardContent>
                      <Typography variant="caption">Max Drawdown</Typography>
                      <Typography variant="h5">{(result.max_drawdown * 100).toFixed(2)}%</Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={6}>
                  <Card>
                    <CardContent>
                      <Typography variant="caption">Win Rate</Typography>
                      <Typography variant="h5">{(result.win_rate * 100).toFixed(1)}%</Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            )}
          </Grid>
        </Grid>
      </Box>
    </AppShell>
  );
}