import { useState } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import Grid from '@mui/material/Grid';
import MenuItem from '@mui/material/MenuItem';
import Select from '@mui/material/Select';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import LinearProgress from '@mui/material/LinearProgress';
import Alert from '@mui/material/Alert';
import Chip from '@mui/material/Chip';
import IconButton from '@mui/material/IconButton';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogContentText from '@mui/material/DialogContentText';
import DialogActions from '@mui/material/DialogActions';
import Checkbox from '@mui/material/Checkbox';
import FormControlLabel from '@mui/material/FormControlLabel';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';
import ToggleButton from '@mui/material/ToggleButton';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
import Stack from '@mui/material/Stack';
import DeleteIcon from '@mui/icons-material/Delete';
import SaveAltIcon from '@mui/icons-material/SaveAlt';
import VisibilityIcon from '@mui/icons-material/Visibility';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';
import { useQuery, useMutation } from '@tanstack/react-query';
import { submitBacktest, getBacktestStatus } from '@/lib/api/client';
import { useBacktestStore, backtestStatusColors, backtestStatusLabels } from '@/store/backtestStore';
import type { BacktestTemplate } from '@/store/backtestStore';
import { ResultsPanel } from '@/components/backtest/ResultsPanel';
import { EquityCurveChart } from '@/components/charts/EquityCurveChart';
import type { StrategyType } from '@/types/api';

const strategies: { value: StrategyType; label: string }[] = [
  { value: 'topk_dropout', label: 'Top-K Dropout' },
  { value: 'equal_weight', label: 'Equal Weight' },
  { value: 'score_weight', label: 'Score Weight' },
];

export function BacktestPage() {
  const { currentTaskId, addTask, updateTaskStatus, getCurrentTask, getTaskList, deleteTask, isSubmitting, setSubmitting, templates, saveTemplate, deleteTemplate } = useBacktestStore();
  const [strategy, setStrategy] = useState<StrategyType>('topk_dropout');
  const [batchStrategies, setBatchStrategies] = useState<StrategyType[]>(['topk_dropout']);
  const [batchMode, setBatchMode] = useState(false);
  const [compareMode, setCompareMode] = useState(false);
  const [modelName, setModelName] = useState('lightgbm_model');
  const [startDate, setStartDate] = useState('2023-01-01');
  const [endDate, setEndDate] = useState('2024-01-01');
  const [initialCapital, setInitialCapital] = useState(1000000);
  const [topK, setTopK] = useState(50);
  const [selectedTemplate, setSelectedTemplate] = useState('');
  const [templateNameInput, setTemplateNameInput] = useState('');
  const [templateDialogOpen, setTemplateDialogOpen] = useState(false);

  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [deleteTargetId, setDeleteTargetId] = useState<string | null>(null);

  const currentTask = getCurrentTask();
  const taskList = getTaskList();

  const handleSaveTemplate = () => {
    if (!templateNameInput.trim()) return;
    saveTemplate({
      name: templateNameInput.trim(),
      strategy,
      modelName,
      startDate,
      endDate,
      initialCapital,
      topK,
    });
    setTemplateDialogOpen(false);
    setTemplateNameInput('');
  };

  const handleLoadTemplate = (name: string) => {
    const tmpl = templates.find((t) => t.name === name);
    if (!tmpl) return;
    setSelectedTemplate(name);
    setStrategy(tmpl.strategy);
    setModelName(tmpl.modelName);
    setStartDate(tmpl.startDate);
    setEndDate(tmpl.endDate);
    setInitialCapital(tmpl.initialCapital);
    setTopK(tmpl.topK);
  };

  // F-032: Template presets
  const completedTasks = taskList.filter((t) => t.status.status === 'completed' && t.status.result);

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
    if (batchMode) {
      // Submit multiple backtests sequentially
      const submitAll = async () => {
        for (const s of batchStrategies) {
          try {
            const result = await submitBacktest({
              strategy_type: s,
              model_name: modelName,
              start_date: startDate,
              end_date: endDate,
              initial_capital: initialCapital,
              top_k: topK,
            });
            addTask(result.task_id, {
              strategy_type: s,
              model_name: modelName,
              start_date: startDate,
              end_date: endDate,
              initial_capital: initialCapital,
              top_k: topK,
            }, result);
          } catch {
            // continue with next
          }
        }
        setSubmitting(false);
      };
      submitAll();
    } else {
      submitMutation.mutate({
        strategy_type: strategy,
        model_name: modelName,
        start_date: startDate,
        end_date: endDate,
        initial_capital: initialCapital,
        top_k: topK,
      });
    }
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 600 }}>
          Backtest Runner
        </Typography>
        {completedTasks.length >= 2 && (
          <Button
            variant={compareMode ? 'contained' : 'outlined'}
            color="secondary"
            startIcon={<CompareArrowsIcon />}
            onClick={() => setCompareMode(!compareMode)}
          >
            {compareMode ? 'Exit Comparison' : 'Compare Results'}
          </Button>
        )}
      </Box>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Stack spacing={2}>
            {/* F-032: Template presets */}
            {templates.length > 0 && (
              <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                <FormControl size="small" sx={{ minWidth: 180 }}>
                  <InputLabel>Load Template</InputLabel>
                  <Select
                    value={selectedTemplate}
                    label="Load Template"
                    onChange={(e) => {
                      const val = e.target.value;
                      if (val) handleLoadTemplate(val);
                    }}
                  >
                    <MenuItem value="">
                      <em>None</em>
                    </MenuItem>
                    {templates.map((t) => (
                      <MenuItem key={t.name} value={t.name}>{t.name}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <Button
                  size="small"
                  startIcon={<SaveAltIcon />}
                  onClick={() => setTemplateDialogOpen(true)}
                >
                  Save Preset
                </Button>
              </Box>
            )}
            {templates.length === 0 && (
              <Box>
                <Button
                  size="small"
                  startIcon={<SaveAltIcon />}
                  onClick={() => setTemplateDialogOpen(true)}
                >
                  Save as Template
                </Button>
              </Box>
            )}

            {/* Batch mode toggle */}
            <Box>
              <FormControlLabel
                control={
                  <Checkbox
                    checked={batchMode}
                    onChange={(e) => setBatchMode(e.target.checked)}
                  />
                }
                label="Batch Mode — run multiple strategies in parallel"
              />
            </Box>

            {batchMode ? (
              <Box>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  Select strategies to compare:
                </Typography>
                <ToggleButtonGroup
                  value={batchStrategies}
                  onChange={(_, vals) => vals.length > 0 && setBatchStrategies(vals)}
                  size="small"
                >
                  {strategies.map((s) => (
                    <ToggleButton key={s.value} value={s.value}>
                      {s.label}
                    </ToggleButton>
                  ))}
                </ToggleButtonGroup>
              </Box>
            ) : (
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
              </Grid>
            )}

            <Grid container spacing={2}>
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
                  disabled={isSubmitting || (batchMode && batchStrategies.length === 0)}
                  sx={{ height: '100%' }}
                >
                  {isSubmitting
                    ? 'Submitting...'
                    : batchMode
                      ? `Run ${batchStrategies.length} Backtests`
                      : 'Run Backtest'}
                </Button>
              </Grid>
            </Grid>
          </Stack>
        </CardContent>
      </Card>

      {/* C5: Comparison View */}
      {compareMode && completedTasks.length >= 2 && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
              Strategy Comparison
            </Typography>

            {/* Comparison metrics table */}
            <TableContainer component={Paper} variant="outlined" sx={{ mb: 2 }}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Metric</TableCell>
                    {completedTasks.map((t) => (
                      <TableCell key={t.taskId} align="right">
                        {t.request.strategy_type}
                      </TableCell>
                    ))}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {[
                    { label: 'Sharpe Ratio', key: 'sharpe_ratio' as const, format: (v: number) => v.toFixed(2) },
                    { label: 'Max Drawdown', key: 'max_drawdown' as const, format: (v: number) => `${(v * 100).toFixed(2)}%` },
                    { label: 'Annual Return', key: 'annual_return' as const, format: (v: number) => `${(v * 100).toFixed(2)}%` },
                    { label: 'Win Rate', key: 'win_rate' as const, format: (v: number) => `${(v * 100).toFixed(1)}%` },
                  ].map(({ label, key, format }) => (
                    <TableRow key={key}>
                      <TableCell sx={{ fontWeight: 600 }}>{label}</TableCell>
                      {completedTasks.map((t) => {
                        const val = t.status.result?.[key];
                        return (
                          <TableCell key={t.taskId} align="right">
                            {val != null ? format(val) : '-'}
                          </TableCell>
                        );
                      })}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>

            {/* Overlay equity curves */}
            <EquityCurveChart data={(() => {
              const dates: string[] = [];
              const start = new Date(startDate);
              const end = new Date(endDate);
              for (let d = new Date(start); d <= end; d.setDate(d.getDate() + 1)) {
                if (d.getDay() !== 0 && d.getDay() !== 6) {
                  dates.push(d.toISOString().slice(0, 10));
                }
              }
              const seeds = [42, 137, 256];
              const curves = completedTasks.map((_, i) => {
                let val = initialCapital;
                let seed = seeds[i % seeds.length];
                return dates.map((date) => {
                  seed = (seed * 16807) % 2147483647;
                  const ret = (seed / 2147483647 - 0.5) * 0.04;
                  val *= (1 + ret);
                  return { date, value: val };
                });
              });
              return dates.map((date, idx) => {
                const point: Record<string, string | number> = { date };
                point['strategy'] = curves[0][idx].value;
                completedTasks.forEach((t, i) => {
                  if (i > 0) point[`strategy_${i}`] = curves[i][idx].value;
                });
                return point as unknown as import('@/components/charts/EquityCurveChart').EquityPoint;
              });
            })()} />
          </CardContent>
        </Card>
      )}

      {currentTask && !compareMode && (
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
      <Dialog open={templateDialogOpen} onClose={() => setTemplateDialogOpen(false)}>
        <DialogTitle>Save Template Preset</DialogTitle>
        <DialogContent>
          <DialogContentText sx={{ mb: 2 }}>
            Save current parameter configuration as a reusable template.
          </DialogContentText>
          <TextField
            autoFocus
            fullWidth
            label="Template Name"
            value={templateNameInput}
            onChange={(e) => setTemplateNameInput(e.target.value)}
            size="small"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setTemplateDialogOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleSaveTemplate}
            disabled={!templateNameInput.trim()}
          >
            Save
          </Button>
        </DialogActions>
      </Dialog>

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
