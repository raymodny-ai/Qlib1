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
import IconButton from '@mui/material/IconButton';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';
import DialogActions from '@mui/material/DialogActions';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
import DeleteIcon from '@mui/icons-material/Delete';
import VisibilityIcon from '@mui/icons-material/Visibility';
import { useQuery, useMutation } from '@tanstack/react-query';
import { submitBacktest, getBacktestStatus } from '@/lib/api/client';
import { useBacktestStore, backtestStatusColors, backtestStatusLabels } from '@/store/backtestStore';
import { ResultsPanel } from '@/components/backtest/ResultsPanel';
import type { StrategyType } from '@/types/api';

const strategies: { value: StrategyType; label: string }[] = [
  { value: 'topk_dropout', label: 'Top-K Dropout' },
  { value: 'equal_weight', label: 'Equal Weight' },
  { value: 'score_weight', label: 'Score Weight' },
];

export function BacktestPage() {
  const { currentTaskId, addTask, updateTaskStatus, getCurrentTask, getTaskList, deleteTask, isSubmitting, setSubmitting } = useBacktestStore();
  const [strategy, setStrategy] = useState<StrategyType>('topk_dropout');
  const [modelName, setModelName] = useState('lightgbm_model');
  const [startDate, setStartDate] = useState('2023-01-01');
  const [endDate, setEndDate] = useState('2024-01-01');
  const [initialCapital, setInitialCapital] = useState(1000000);
  const [topK, setTopK] = useState(50);

  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [deleteTargetId, setDeleteTargetId] = useState<string | null>(null);

  const currentTask = getCurrentTask();
  const taskList = getTaskList();

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
              <ResultsPanel
                result={currentTask.status}
                startDate={startDate}
                endDate={endDate}
                initialCapital={initialCapital}
              />
            )}
          </CardContent>
        </Card>
      )}

      {/* Task History */}
      {taskList.length > 0 && (
        <Card>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
              Task History
            </Typography>
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Task ID</TableCell>
                    <TableCell>Strategy</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Submitted</TableCell>
                    <TableCell align="right">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {taskList.map((task) => (
                    <TableRow
                      key={task.taskId}
                      hover
                      sx={{ cursor: 'pointer', bgcolor: task.taskId === currentTaskId ? 'action.selected' : 'inherit' }}
                      onClick={() => {
                        useBacktestStore.getState().setCurrentTask(task.taskId);
                        if (task.status.status === 'running') {
                          pollStatus();
                        }
                      }}
                    >
                      <TableCell sx={{ fontFamily: 'monospace', fontSize: 12 }}>
                        {task.taskId.slice(0, 12)}...
                      </TableCell>
                      <TableCell>{task.request.strategy_type}</TableCell>
                      <TableCell>
                        <Chip
                          label={backtestStatusLabels[task.status.status]}
                          sx={{ bgcolor: backtestStatusColors[task.status.status], color: '#fff' }}
                          size="small"
                        />
                      </TableCell>
                      <TableCell sx={{ fontSize: 12, color: 'text.secondary' }}>
                        {new Date(task.submittedAt).toLocaleString()}
                      </TableCell>
                      <TableCell align="right">
                        <IconButton
                          size="small"
                          onClick={(e) => {
                            e.stopPropagation();
                            useBacktestStore.getState().setCurrentTask(task.taskId);
                          }}
                        >
                          <VisibilityIcon fontSize="small" />
                        </IconButton>
                        <IconButton
                          size="small"
                          onClick={(e) => {
                            e.stopPropagation();
                            setDeleteTargetId(task.taskId);
                            setDeleteConfirmOpen(true);
                          }}
                        >
                          <DeleteIcon fontSize="small" />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      )}

      {/* Delete confirmation dialog */}
      <Dialog open={deleteConfirmOpen} onClose={() => setDeleteConfirmOpen(false)}>
        <DialogTitle>Delete Task</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete task {deleteTargetId?.slice(0, 12)}...? This action cannot be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteConfirmOpen(false)}>Cancel</Button>
          <Button
            color="error"
            onClick={() => {
              if (deleteTargetId) {
                deleteTask(deleteTargetId);
              }
              setDeleteConfirmOpen(false);
              setDeleteTargetId(null);
            }}
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
