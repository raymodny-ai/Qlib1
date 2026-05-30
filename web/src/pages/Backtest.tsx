import { useState } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import Grid from '@mui/material/Grid';
import MenuItem from '@mui/material/MenuItem';
import LinearProgress from '@mui/material/LinearProgress';
import Alert from '@mui/material/Alert';
import Chip from '@mui/material/Chip';
import { useQuery, useMutation } from '@tanstack/react-query';
import { submitBacktest, getBacktestStatus } from '@/lib/api/client';
import { useBacktestStore, backtestStatusColors, backtestStatusLabels } from '@/store/backtestStore';
import type { StrategyType } from '@/types/api';

const strategies: { value: StrategyType; label: string }[] = [
  { value: 'topk_dropout', label: 'Top-K Dropout' },
  { value: 'equal_weight', label: 'Equal Weight' },
  { value: 'score_weight', label: 'Score Weight' },
];

export function BacktestPage() {
  const { currentTaskId, addTask, updateTaskStatus, getCurrentTask, isSubmitting, setSubmitting } = useBacktestStore();
  const [strategy, setStrategy] = useState<StrategyType>('topk_dropout');
  const [modelName, setModelName] = useState('lightgbm_model');
  const [startDate, setStartDate] = useState('2023-01-01');
  const [endDate, setEndDate] = useState('2024-01-01');
  const [initialCapital, setInitialCapital] = useState(1000000);
  const [topK, setTopK] = useState(50);

  const currentTask = getCurrentTask();

  const { data: taskStatus, refetch: pollStatus } = useQuery({
    queryKey: ['backtest-status', currentTaskId],
    queryFn: () => getBacktestStatus(currentTaskId!),
    enabled: !!currentTaskId && currentTask?.status.status === 'running',
    refetchInterval: 2000,
  });

  if (taskStatus && currentTaskId) {
    updateTaskStatus(currentTaskId, taskStatus);
  }

  const submitMutation = useMutation({
    mutationFn: submitBacktest,
    onSuccess: (data) => {
      addTask(data.task_id, {
        strategy_type: strategy,
        model_name: modelName,
        start_date: startDate,
        end_date: endDate,
        initial_capital: initialCapital,
        top_k: topK,
      }, data);
      setSubmitting(false);
      pollStatus();
    },
    onError: () => {
      setSubmitting(false);
    },
  });

  const handleSubmit = () => {
    setSubmitting(true);
    submitMutation.mutate({
      strategy_type: strategy,
      model_name: modelName,
      start_date: startDate,
      end_date: endDate,
      initial_capital: initialCapital,
      top_k: topK,
    });
  };

  return (
    <Box>
      <Typography variant="h4" sx={{ mb: 3, fontWeight: 600 }}>
        Backtest Runner
      </Typography>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Grid container spacing={2}>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                select
                label="Strategy"
                value={strategy}
                onChange={(e) => setStrategy(e.target.value as StrategyType)}
                size="small"
              >
                {strategies.map((s) => (
                  <MenuItem key={s.value} value={s.value}>{s.label}</MenuItem>
                ))}
              </TextField>
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="Model Name"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                size="small"
              />
            </Grid>
            <Grid item xs={6} md={2}>
              <TextField
                fullWidth
                label="Start Date"
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                size="small"
                InputLabelProps={{ shrink: true }}
              />
            </Grid>
            <Grid item xs={6} md={2}>
              <TextField
                fullWidth
                label="End Date"
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                size="small"
                InputLabelProps={{ shrink: true }}
              />
            </Grid>
            <Grid item xs={6} md={4}>
              <TextField
                fullWidth
                label="Initial Capital"
                type="number"
                value={initialCapital}
                onChange={(e) => setInitialCapital(Number(e.target.value))}
                size="small"
              />
            </Grid>
            <Grid item xs={6} md={4}>
              <TextField
                fullWidth
                label="Top K"
                type="number"
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                size="small"
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <Button
                variant="contained"
                fullWidth
                onClick={handleSubmit}
                disabled={isSubmitting}
                sx={{ height: '100%' }}
              >
                {isSubmitting ? 'Submitting...' : 'Run Backtest'}
              </Button>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {currentTask && (
        <Card>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 2 }}>
              Task: {currentTask.taskId}
            </Typography>
            <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
              <Chip
                label={backtestStatusLabels[currentTask.status.status]}
                sx={{ bgcolor: backtestStatusColors[currentTask.status.status], color: '#fff' }}
                size="small"
              />
              {currentTask.status.status === 'running' && (
                <Box sx={{ flexGrow: 1 }}>
                  <LinearProgress variant="determinate" value={currentTask.status.progress * 100} />
                </Box>
              )}
            </Box>
            {currentTask.status.error && (
              <Alert severity="error">{currentTask.status.error}</Alert>
            )}
            {currentTask.status.result && (
              <Grid container spacing={2}>
                {Object.entries(currentTask.status.result).map(([key, value]) => (
                  <Grid item xs={6} md={3} key={key}>
                    <Typography variant="caption" color="text.secondary">
                      {key.replace(/_/g, ' ')}
                    </Typography>
                    <Typography variant="h6">
                      {typeof value === 'number' ? value.toFixed(4) : String(value)}
                    </Typography>
                  </Grid>
                ))}
              </Grid>
            )}
          </CardContent>
        </Card>
      )}
    </Box>
  );
}
