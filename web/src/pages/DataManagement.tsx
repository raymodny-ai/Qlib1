import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Button from '@mui/material/Button';
import Grid from '@mui/material/Grid';
import Chip from '@mui/material/Chip';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
import LinearProgress from '@mui/material/LinearProgress';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import TextField from '@mui/material/TextField';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';
import ToggleButton from '@mui/material/ToggleButton';
import Tooltip from '@mui/material/Tooltip';
import Stack from '@mui/material/Stack';
import CircularProgress from '@mui/material/CircularProgress';
import CloudDoneIcon from '@mui/icons-material/CloudDone';
import CloudOffIcon from '@mui/icons-material/CloudOff';
import CloudQueueIcon from '@mui/icons-material/CloudQueue';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import RefreshIcon from '@mui/icons-material/Refresh';
import StorageIcon from '@mui/icons-material/Storage';
import { useState } from 'react';
import { useSnackbar } from 'notistack';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  getDataSources,
  getDatasets,
  triggerDataIngest,
  previewDataset as fetchPreviewDataset,
  getInstruments,
} from '@/lib/api/client';
import type { DataSourceInfo, DatasetInfo, IngestMode } from '@/types/api';

export function DataManagementPage() {
  const queryClient = useQueryClient();
  const { enqueueSnackbar } = useSnackbar();

  // Ingest dialog state
  const [ingestOpen, setIngestOpen] = useState(false);
  const [ingestDataset, setIngestDataset] = useState('alpha158');
  const [ingestMode, setIngestMode] = useState<IngestMode>('incremental');
  const [ingesting, setIngesting] = useState(false);
  const [ingestProgress, setIngestProgress] = useState<string | null>(null);

  // Preview dialog state
  const [previewOpen, setPreviewOpen] = useState(false);
  const [previewDataset, setPreviewDataset] = useState('alpha158');

  const { data: instruments, isLoading } = useQuery({
    queryKey: ['instruments'],
    queryFn: () => getInstruments({ limit: 50 }),
  });

  const { data: sources } = useQuery({
    queryKey: ['data-sources'],
    queryFn: getDataSources,
    refetchInterval: 60000,
  });

  const { data: datasets } = useQuery({
    queryKey: ['datasets'],
    queryFn: getDatasets,
  });

  const { data: preview, refetch: refetchPreview } = useQuery({
    queryKey: ['dataset-preview', previewDataset],
    queryFn: () => fetchPreviewDataset(previewDataset, { limit: 50 }),
    enabled: false,
  });

  const handleIngest = async () => {
    setIngesting(true);
    setIngestProgress('Submitting...');
    try {
      const result = await triggerDataIngest({
        dataset: ingestDataset,
        mode: ingestMode,
        force: false,
      });
      setIngestProgress(result.message);
      enqueueSnackbar(`Ingestion started: ${result.message}`, { variant: 'success' });
      queryClient.invalidateQueries({ queryKey: ['datasets'] });
      queryClient.invalidateQueries({ queryKey: ['data-sources'] });
      setTimeout(() => {
        setIngestOpen(false);
        setIngestProgress(null);
      }, 2000);
    } catch {
      enqueueSnackbar('Failed to trigger ingestion', { variant: 'error' });
    } finally {
      setIngesting(false);
    }
  };

  const handlePreview = (dataset: string) => {
    setPreviewDataset(dataset);
    setPreviewOpen(true);
    setTimeout(() => refetchPreview(), 100);
  };

  const statusIcon = (status: string) => {
    switch (status) {
      case 'connected': return <CloudDoneIcon color="success" />;
      case 'disconnected': return <CloudOffIcon color="error" />;
      case 'degraded': return <CloudQueueIcon color="warning" />;
      default: return <ErrorOutlineIcon color="error" />;
    }
  };

  const statusColor = (status: string): 'success' | 'error' | 'warning' | 'default' => {
    switch (status) {
      case 'connected': return 'success';
      case 'disconnected': return 'error';
      case 'degraded': return 'warning';
      default: return 'error';
    }
  };

  const qualityColor = (score: number) => {
    if (score >= 90) return 'success.main';
    if (score >= 70) return 'warning.main';
    return 'error.main';
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 600 }}>
          Data Management
        </Typography>
        <Button
          variant="contained"
          startIcon={<RefreshIcon />}
          onClick={() => {
            queryClient.invalidateQueries({ queryKey: ['data-sources'] });
            queryClient.invalidateQueries({ queryKey: ['datasets'] });
            queryClient.invalidateQueries({ queryKey: ['instruments'] });
          }}
        >
          Refresh All
        </Button>
      </Box>

      {/* Data Source Cards */}
      <Typography variant="h5" sx={{ mb: 2 }}>
        Data Sources
      </Typography>
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {sources?.map((src: DataSourceInfo) => (
          <Grid item xs={12} sm={6} md={3} key={src.source_id}>
            <Card
              sx={{
                borderLeft: 4,
                borderColor: statusColor(src.status) + '.main',
              }}
            >
              <CardContent>
                <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1 }}>
                  {statusIcon(src.status)}
                  <Typography variant="h6" sx={{ fontSize: '1rem' }}>
                    {src.name}
                  </Typography>
                </Stack>
                <Chip
                  label={src.status.toUpperCase()}
                  color={statusColor(src.status)}
                  size="small"
                  sx={{ mb: 1 }}
                />
                <Typography variant="body2" color="text.secondary">
                  {src.description}
                </Typography>
                <Stack direction="row" justifyContent="space-between" sx={{ mt: 1.5 }}>
                  <Typography variant="caption" color="text.secondary">
                    Records: {src.record_count.toLocaleString()}
                  </Typography>
                  <Tooltip title={`Quality Score: ${src.quality_score}/100`}>
                    <Typography
                      variant="caption"
                      sx={{ fontWeight: 600, color: qualityColor(src.quality_score) }}
                    >
                      {src.quality_score.toFixed(1)}%
                    </Typography>
                  </Tooltip>
                </Stack>
                {src.last_sync && (
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
                    Last sync: {new Date(src.last_sync).toLocaleString()}
                  </Typography>
                )}
              </CardContent>
            </Card>
          </Grid>
        ))}
        {!sources && (
          <Grid item xs={12}>
            <LinearProgress />
          </Grid>
        )}
      </Grid>

      {/* Datasets Table + Ingest */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
        <Typography variant="h5">
          Datasets
          {datasets && (
            <Chip label={`${datasets.length} datasets`} size="small" sx={{ ml: 1 }} />
          )}
        </Typography>
        <Button
          variant="contained"
          onClick={() => setIngestOpen(true)}
          startIcon={<StorageIcon />}
        >
          Ingest Data
        </Button>
      </Box>

      <TableContainer component={Paper} sx={{ mb: 4 }}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Dataset</TableCell>
              <TableCell>Description</TableCell>
              <TableCell align="right">Instruments</TableCell>
              <TableCell align="right">Fields</TableCell>
              <TableCell>Date Range</TableCell>
              <TableCell align="right">Size</TableCell>
              <TableCell>Last Updated</TableCell>
              <TableCell>Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {datasets?.map((ds: DatasetInfo) => (
              <TableRow key={ds.name}>
                <TableCell>
                  <Chip label={ds.name} size="small" color="primary" variant="outlined" />
                </TableCell>
                <TableCell>{ds.description}</TableCell>
                <TableCell align="right">{ds.n_instruments.toLocaleString()}</TableCell>
                <TableCell align="right">{ds.n_fields}</TableCell>
                <TableCell>
                  {ds.date_range.start} → {ds.date_range.end}
                </TableCell>
                <TableCell align="right">{ds.size_mb.toFixed(1)} MB</TableCell>
                <TableCell>
                  {ds.last_updated
                    ? new Date(ds.last_updated).toLocaleDateString()
                    : '-'}
                </TableCell>
                <TableCell>
                  <Button size="small" variant="outlined" onClick={() => handlePreview(ds.name)}>
                    Preview
                  </Button>
                </TableCell>
              </TableRow>
            ))}
            {!datasets && (
              <TableRow>
                <TableCell colSpan={8}>
                  <LinearProgress />
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Instruments Table */}
      <Typography variant="h5" sx={{ mb: 2 }}>
        Instruments
        {instruments && (
          <Chip label={`${instruments.length} shown`} size="small" sx={{ ml: 1 }} />
        )}
      </Typography>
      <TableContainer component={Paper}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Symbol</TableCell>
              <TableCell>Name</TableCell>
              <TableCell>Sector</TableCell>
              <TableCell>Industry</TableCell>
              <TableCell align="right">Market Cap</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {instruments?.map((inst) => (
              <TableRow key={inst.symbol}>
                <TableCell>
                  <Chip label={inst.symbol} size="small" color="primary" variant="outlined" />
                </TableCell>
                <TableCell>{inst.name || '-'}</TableCell>
                <TableCell>{inst.sector || '-'}</TableCell>
                <TableCell>{inst.industry || '-'}</TableCell>
                <TableCell align="right">
                  {inst.market_cap ? `$${(inst.market_cap / 1e9).toFixed(1)}B` : '-'}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Ingest Dialog */}
      <Dialog open={ingestOpen} onClose={() => !ingesting && setIngestOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Trigger Data Ingestion</DialogTitle>
        <DialogContent>
          <Stack spacing={3} sx={{ mt: 1 }}>
            <FormControl fullWidth>
              <InputLabel>Dataset</InputLabel>
              <Select
                value={ingestDataset}
                label="Dataset"
                onChange={(e) => setIngestDataset(e.target.value)}
              >
                {(datasets || []).map((ds) => (
                  <MenuItem key={ds.name} value={ds.name}>
                    {ds.name} — {ds.description}
                  </MenuItem>
                ))}
                {!datasets && [
                  <MenuItem key="alpha158" value="alpha158">alpha158</MenuItem>,
                  <MenuItem key="alpha360" value="alpha360">alpha360</MenuItem>,
                  <MenuItem key="alpha101" value="alpha101">alpha101</MenuItem>,
                  <MenuItem key="fundamentals" value="fundamentals">fundamentals</MenuItem>,
                ]}
              </Select>
            </FormControl>
            <Box>
              <Typography variant="body2" sx={{ mb: 1 }}>Mode</Typography>
              <ToggleButtonGroup
                value={ingestMode}
                exclusive
                onChange={(_, val) => val && setIngestMode(val)}
                size="small"
                fullWidth
              >
                <ToggleButton value="incremental">Incremental</ToggleButton>
                <ToggleButton value="full">Full Reload</ToggleButton>
              </ToggleButtonGroup>
            </Box>
            {ingestProgress && (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                {ingesting && <CircularProgress size={18} />}
                <Typography variant="body2" color="text.secondary">
                  {ingestProgress}
                </Typography>
              </Box>
            )}
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setIngestOpen(false)} disabled={ingesting}>
            Cancel
          </Button>
          <Button
            variant="contained"
            onClick={handleIngest}
            disabled={ingesting}
            startIcon={ingesting ? <CircularProgress size={18} /> : undefined}
          >
            {ingesting ? 'Ingesting...' : 'Start Ingestion'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Preview Dialog */}
      <Dialog
        open={previewOpen}
        onClose={() => setPreviewOpen(false)}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>
          Dataset Preview: {previewDataset}
          {preview && (
            <Chip
              label={`${preview.preview_rows} of ${preview.total_rows.toLocaleString()} rows`}
              size="small"
              sx={{ ml: 1 }}
            />
          )}
        </DialogTitle>
        <DialogContent>
          {!preview ? (
            <LinearProgress />
          ) : (
            <TableContainer sx={{ maxHeight: 400 }}>
              <Table size="small" stickyHeader>
                <TableHead>
                  <TableRow>
                    {preview.columns.map((col: string) => (
                      <TableCell key={col} sx={{ fontWeight: 600 }}>
                        {col}
                      </TableCell>
                    ))}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {preview.rows.map((row: Record<string, unknown>, i: number) => (
                    <TableRow key={i}>
                      {preview.columns.map((col: string) => (
                        <TableCell key={col} sx={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>
                          {row[col] != null ? String(row[col]) : '-'}
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setPreviewOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
