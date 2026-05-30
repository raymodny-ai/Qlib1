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
import Alert from '@mui/material/Alert';
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
import TablePagination from '@mui/material/TablePagination';
import Stack from '@mui/material/Stack';
import CircularProgress from '@mui/material/CircularProgress';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import DownloadIcon from '@mui/icons-material/Download';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import { useState, useMemo } from 'react';
import { useSnackbar } from 'notistack';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  getComplianceStatus,
  generateSOXReport,
  downloadSOXReport,
  getAuditLogs,
  verifyAuditChain,
  exportAuditReport,
} from '@/lib/api/client';
import type { AuditLogEntry } from '@/types/api';

export function CompliancePage() {
  const queryClient = useQueryClient();
  const { enqueueSnackbar } = useSnackbar();

  // SOX report generation modal
  const [soxModalOpen, setSoxModalOpen] = useState(false);
  const [soxReportData, setSoxReportData] = useState<Record<string, unknown> | null>(null);
  const [soxGenerating, setSoxGenerating] = useState(false);

  // Audit log filters & pagination
  const [auditPage, setAuditPage] = useState(0);
  const [auditRowsPerPage, setAuditRowsPerPage] = useState(20);
  const [auditEventType, setAuditEventType] = useState('');
  const [auditUserFilter, setAuditUserFilter] = useState('');
  const [auditStartTime, setAuditStartTime] = useState('');
  const [auditEndTime, setAuditEndTime] = useState('');

  const { data: compliance } = useQuery({
    queryKey: ['compliance-status'],
    queryFn: getComplianceStatus,
  });

  const { data: auditChain } = useQuery({
    queryKey: ['audit-chain'],
    queryFn: () => verifyAuditChain(),
  });

  const auditParams = useMemo(() => ({
    limit: auditRowsPerPage,
    event_type: auditEventType || undefined,
    user: auditUserFilter || undefined,
    start_time: auditStartTime || undefined,
    end_time: auditEndTime || undefined,
  }), [auditRowsPerPage, auditEventType, auditUserFilter, auditStartTime, auditEndTime]);

  const { data: auditLogs } = useQuery({
    queryKey: ['audit-logs', auditParams],
    queryFn: () => getAuditLogs(auditParams),
  });

  const handleGenerateSOX = async () => {
    setSoxGenerating(true);
    try {
      const report = await generateSOXReport({});
      setSoxReportData(report as unknown as Record<string, unknown>);
      setSoxModalOpen(true);
      queryClient.invalidateQueries({ queryKey: ['compliance-status'] });
      enqueueSnackbar('SOX report generated successfully', { variant: 'success' });
    } catch {
      enqueueSnackbar('Failed to generate SOX report', { variant: 'error' });
    } finally {
      setSoxGenerating(false);
    }
  };

  const handleDownloadSOX = async () => {
    try {
      const report = await downloadSOXReport({});
      const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `sox_report_${report.quarter || 'current'}_${new Date().toISOString().slice(0, 10)}.json`;
      a.click();
      URL.revokeObjectURL(url);
      enqueueSnackbar('SOX report downloaded', { variant: 'success' });
    } catch {
      enqueueSnackbar('Failed to download SOX report', { variant: 'error' });
    }
  };

  const handleExportAudit = async () => {
    try {
      await exportAuditReport(auditParams);
      enqueueSnackbar('Audit report exported', { variant: 'success' });
    } catch {
      enqueueSnackbar('Failed to export audit report', { variant: 'error' });
    }
  };

  // Compute trend indicators
  const controlsPassed = compliance?.controls?.filter((c) => c.status === 'passed').length ?? 0;
  const controlsWarning = compliance?.controls?.filter((c) => c.status === 'warning').length ?? 0;
  const controlsFailed = compliance?.controls?.filter((c) => c.status === 'failed').length ?? 0;
  const totalControls = compliance?.controls?.length ?? 0;

  // Audit chain visualization data
  const chainEntries = auditLogs?.entries?.slice(0, 10) ?? [];

  return (
    <Box>
      <Typography variant="h4" sx={{ mb: 3, fontWeight: 600 }}>
        Compliance & Audit
      </Typography>

      {/* KPI Cards (C2) */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1 }}>
                <AccessTimeIcon color="action" />
                <Typography variant="body2" color="text.secondary">Overall Status</Typography>
              </Stack>
              {compliance && (
                <Chip
                  label={compliance.overall_status.toUpperCase()}
                  color={
                    compliance.overall_status === 'compliant'
                      ? 'success'
                      : compliance.overall_status === 'warning'
                      ? 'warning'
                      : 'error'
                  }
                  sx={{ fontWeight: 600 }}
                />
              )}
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1 }}>
                <TrendingUpIcon color="success" />
                <Typography variant="body2" color="text.secondary">Passed Controls</Typography>
              </Stack>
              <Typography variant="h4" sx={{ fontWeight: 700, color: 'success.main' }}>
                {controlsPassed}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                of {totalControls} total controls
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1 }}>
                <TrendingDownIcon color="warning" />
                <Typography variant="body2" color="text.secondary">Warnings</Typography>
              </Stack>
              <Typography variant="h4" sx={{ fontWeight: 700, color: 'warning.main' }}>
                {controlsWarning}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {controlsWarning > 0 ? 'Needs attention' : 'No warnings'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 1 }}>
                <TrendingDownIcon color="error" />
                <Typography variant="body2" color="text.secondary">Failed</Typography>
              </Stack>
              <Typography variant="h4" sx={{ fontWeight: 700, color: 'error.main' }}>
                {controlsFailed}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {controlsFailed > 0 ? 'Requires immediate action' : 'No failures'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Audit Chain + SOX Actions */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 1 }}>
                Audit Chain
              </Typography>
              {auditChain && (
                <>
                  <Chip
                    label={auditChain.verified ? 'VERIFIED' : 'BROKEN'}
                    color={auditChain.verified ? 'success' : 'error'}
                    sx={{ mb: 1 }}
                  />
                  <Typography variant="body2" color="text.secondary">
                    {auditChain.total_entries} entries on {auditChain.date}
                  </Typography>
                  {auditChain.broken_at != null && (
                    <Alert severity="error" sx={{ mt: 1 }}>
                      Chain broken at entry #{auditChain.broken_at}
                    </Alert>
                  )}
                </>
              )}
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>
                SOX Report
              </Typography>
              <Stack spacing={1}>
                <Button
                  variant="contained"
                  onClick={handleGenerateSOX}
                  disabled={soxGenerating}
                  fullWidth
                  startIcon={soxGenerating ? <CircularProgress size={18} /> : undefined}
                >
                  {soxGenerating ? 'Generating...' : 'Generate SOX Report'}
                </Button>
                {soxReportData && (
                  <Button
                    variant="outlined"
                    onClick={() => setSoxModalOpen(true)}
                    fullWidth
                  >
                    View Last Report
                  </Button>
                )}
                <Button
                  variant="outlined"
                  onClick={handleDownloadSOX}
                  fullWidth
                  startIcon={<DownloadIcon />}
                >
                  Download as JSON
                </Button>
              </Stack>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 1 }}>
                Last Audit
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Period: {compliance?.period ?? 'N/A'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Chain Verified: {compliance?.audit_chain_verified ? 'Yes' : 'No'}
              </Typography>
              {compliance?.controls && (
                <Typography variant="body2" color="text.secondary">
                  Controls: {totalControls} ({controlsPassed} passed)
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Controls Table */}
      {compliance?.controls && (
        <>
          <Typography variant="h5" sx={{ mb: 2 }}>
            Controls ({totalControls})
          </Typography>
          <TableContainer component={Paper} sx={{ mb: 3 }}>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Control ID</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Details</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {compliance.controls.map((ctrl) => (
                  <TableRow key={ctrl.control_id}>
                    <TableCell>{ctrl.control_id}</TableCell>
                    <TableCell>
                      <Chip
                        label={ctrl.status}
                        color={ctrl.status === 'passed' ? 'success' : ctrl.status === 'warning' ? 'warning' : 'error'}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>{ctrl.details || '-'}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </>
      )}

      {/* Audit Chain Visualization (C7) */}
      {chainEntries.length > 0 && auditChain && (
        <>
          <Typography variant="h5" sx={{ mb: 2 }}>
            Audit Chain Visualization
          </Typography>
          <Paper sx={{ p: 3, mb: 3, overflow: 'auto' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0, minWidth: chainEntries.length * 100 }}>
              {chainEntries.map((entry: AuditLogEntry, idx: number) => {
                const isBroken = auditChain.broken_at != null && idx >= auditChain.broken_at;
                return (
                  <Box key={entry.event_id} sx={{ display: 'flex', alignItems: 'center' }}>
                    <Box
                      sx={{
                        width: 56,
                        height: 56,
                        borderRadius: '50%',
                        bgcolor: isBroken ? 'error.main' : 'success.main',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: '#fff',
                        fontSize: '0.6rem',
                        fontWeight: 600,
                        flexShrink: 0,
                      }}
                      title={`${entry.event_type} — ${entry.user_id}`}
                    >
                      <span>#{idx + 1}</span>
                      <span style={{ fontSize: '0.5rem', opacity: 0.8 }}>
                        {entry.hash ? entry.hash.slice(0, 4) : '...'}
                      </span>
                    </Box>
                    {idx < chainEntries.length - 1 && (
                      <Box
                        sx={{
                          width: 44,
                          height: 3,
                          bgcolor: isBroken ? 'error.light' : 'success.light',
                        }}
                      />
                    )}
                  </Box>
                );
              })}
            </Box>
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block', textAlign: 'center' }}>
              Each node = hash entry. Green = verified. Red = broken.
            </Typography>
          </Paper>
        </>
      )}

      {/* Audit Logs (C1) */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
        <Typography variant="h5">
          Audit Logs
          {auditLogs && (
            <Chip label={`${auditLogs.total} entries`} size="small" sx={{ ml: 1 }} />
          )}
        </Typography>
        <Button variant="outlined" size="small" onClick={handleExportAudit}>
          Export
        </Button>
      </Box>

      {/* Audit Filters */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} alignItems="flex-start">
          <FormControl size="small" sx={{ minWidth: 140 }}>
            <InputLabel>Event Type</InputLabel>
            <Select
              value={auditEventType}
              label="Event Type"
              onChange={(e) => { setAuditEventType(e.target.value); setAuditPage(0); }}
            >
              <MenuItem value="">All</MenuItem>
              <MenuItem value="gate_action">Gate Action</MenuItem>
              <MenuItem value="login">Login</MenuItem>
              <MenuItem value="permission_change">Permission Change</MenuItem>
              <MenuItem value="data_access">Data Access</MenuItem>
              <MenuItem value="system_config">System Config</MenuItem>
            </Select>
          </FormControl>
          <TextField
            label="User"
            size="small"
            value={auditUserFilter}
            onChange={(e) => { setAuditUserFilter(e.target.value); setAuditPage(0); }}
            sx={{ minWidth: 140 }}
          />
          <TextField
            label="Start Time"
            type="datetime-local"
            size="small"
            value={auditStartTime}
            onChange={(e) => { setAuditStartTime(e.target.value); setAuditPage(0); }}
            InputLabelProps={{ shrink: true }}
          />
          <TextField
            label="End Time"
            type="datetime-local"
            size="small"
            value={auditEndTime}
            onChange={(e) => { setAuditEndTime(e.target.value); setAuditPage(0); }}
            InputLabelProps={{ shrink: true }}
          />
        </Stack>
      </Paper>

      <TableContainer component={Paper}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Event ID</TableCell>
              <TableCell>Type</TableCell>
              <TableCell>User</TableCell>
              <TableCell>Details</TableCell>
              <TableCell>Hash</TableCell>
              <TableCell>Timestamp</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {auditLogs?.entries.map((entry) => (
              <TableRow key={entry.event_id}>
                <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>
                  {entry.event_id.slice(0, 12)}...
                </TableCell>
                <TableCell>
                  <Chip label={entry.event_type} size="small" variant="outlined" />
                </TableCell>
                <TableCell>{entry.user_id}</TableCell>
                <TableCell sx={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {JSON.stringify(entry.details)}
                </TableCell>
                <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.7rem' }}>
                  {entry.hash ? `${entry.hash.slice(0, 8)}...` : '-'}
                </TableCell>
                <TableCell>{new Date(entry.timestamp).toLocaleString()}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {auditLogs && (
        <TablePagination
          component="div"
          count={auditLogs.total}
          page={auditPage}
          onPageChange={(_, page) => setAuditPage(page)}
          rowsPerPage={auditRowsPerPage}
          onRowsPerPageChange={(e) => {
            setAuditRowsPerPage(parseInt(e.target.value, 10));
            setAuditPage(0);
          }}
          rowsPerPageOptions={[10, 20, 50, 100]}
        />
      )}

      {/* SOX Report Modal (B1) */}
      <Dialog
        open={soxModalOpen}
        onClose={() => setSoxModalOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>SOX Compliance Report</DialogTitle>
        <DialogContent>
          {soxReportData && (
            <>
              <Typography variant="subtitle1" sx={{ mb: 1 }}>
                Quarter: {(soxReportData as Record<string, unknown>).quarter as string || 'Current'}
              </Typography>
              <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 2 }}>
                Generated: {(soxReportData as Record<string, unknown>).generated_at as string}
              </Typography>

              {(soxReportData as Record<string, unknown>).summary && (
                <Paper variant="outlined" sx={{ p: 2, mb: 2 }}>
                  <Typography variant="subtitle1" sx={{ mb: 1, fontWeight: 600 }}>
                    Summary
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={3}>
                      <Typography variant="body2">Total Controls</Typography>
                      <Typography variant="h6">
                        {((soxReportData as Record<string, unknown>).summary as Record<string, number>).total_controls}
                      </Typography>
                    </Grid>
                    <Grid item xs={3}>
                      <Typography variant="body2" color="success.main">Passed</Typography>
                      <Typography variant="h6" color="success.main">
                        {((soxReportData as Record<string, unknown>).summary as Record<string, number>).passed}
                      </Typography>
                    </Grid>
                    <Grid item xs={3}>
                      <Typography variant="body2" color="warning.main">Warnings</Typography>
                      <Typography variant="h6" color="warning.main">
                        {((soxReportData as Record<string, unknown>).summary as Record<string, number>).warnings}
                      </Typography>
                    </Grid>
                    <Grid item xs={3}>
                      <Typography variant="body2" color="error.main">Failed</Typography>
                      <Typography variant="h6" color="error.main">
                        {((soxReportData as Record<string, unknown>).summary as Record<string, number>).failed}
                      </Typography>
                    </Grid>
                  </Grid>
                </Paper>
              )}

              <Typography variant="subtitle1" sx={{ mb: 1, fontWeight: 600 }}>
                Controls Detail
              </Typography>
              <TableContainer component={Paper} variant="outlined">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Control ID</TableCell>
                      <TableCell>Status</TableCell>
                      <TableCell>Details</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {((soxReportData as Record<string, unknown>).controls as Array<Record<string, unknown>>)?.map(
                      (ctrl: Record<string, unknown>, i: number) => (
                        <TableRow key={i}>
                          <TableCell>{ctrl.control_id as string}</TableCell>
                          <TableCell>
                            <Chip
                              label={ctrl.status as string}
                              color={
                                ctrl.status === 'passed'
                                  ? 'success'
                                  : ctrl.status === 'warning'
                                  ? 'warning'
                                  : 'error'
                              }
                              size="small"
                            />
                          </TableCell>
                          <TableCell>{(ctrl.details as string) || '-'}</TableCell>
                        </TableRow>
                      )
                    )}
                  </TableBody>
                </Table>
              </TableContainer>
            </>
          )}
          {!soxReportData && <LinearProgress />}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSoxModalOpen(false)}>Close</Button>
          <Button
            variant="contained"
            onClick={handleDownloadSOX}
            startIcon={<DownloadIcon />}
          >
            Download JSON
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
