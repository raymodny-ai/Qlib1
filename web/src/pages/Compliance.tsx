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
import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  getComplianceStatus,
  generateSOXReport,
  getAuditLogs,
  verifyAuditChain,
} from '@/lib/api/client';

export function CompliancePage() {
  const queryClient = useQueryClient();

  const { data: compliance } = useQuery({
    queryKey: ['compliance-status'],
    queryFn: getComplianceStatus,
  });

  const { data: auditChain } = useQuery({
    queryKey: ['audit-chain'],
    queryFn: () => verifyAuditChain(),
  });

  const { data: auditLogs } = useQuery({
    queryKey: ['audit-logs'],
    queryFn: () => getAuditLogs({ limit: 20 }),
  });

  const handleGenerateSOX = async () => {
    const report = await generateSOXReport({});
    queryClient.invalidateQueries({ queryKey: ['compliance-status'] });
  };

  return (
    <Box>
      <Typography variant="h4" sx={{ mb: 3, fontWeight: 600 }}>
        Compliance & Audit
      </Typography>

      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 1 }}>
                Overall Status
              </Typography>
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
                />
              )}
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 1 }}>
                Audit Chain
              </Typography>
              {auditChain && (
                <Chip
                  label={auditChain.verified ? 'VERIFIED' : 'BROKEN'}
                  color={auditChain.verified ? 'success' : 'error'}
                />
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
              <Button variant="contained" onClick={handleGenerateSOX} fullWidth>
                Generate SOX Report
              </Button>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {compliance?.controls && (
        <>
          <Typography variant="h5" sx={{ mb: 2 }}>
            Controls
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

      <Typography variant="h5" sx={{ mb: 2 }}>
        Recent Audit Logs
      </Typography>
      <TableContainer component={Paper}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Event ID</TableCell>
              <TableCell>Type</TableCell>
              <TableCell>User</TableCell>
              <TableCell>Timestamp</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {auditLogs?.entries.map((entry) => (
              <TableRow key={entry.event_id}>
                <TableCell>{entry.event_id}</TableCell>
                <TableCell>{entry.event_type}</TableCell>
                <TableCell>{entry.user_id}</TableCell>
                <TableCell>{new Date(entry.timestamp).toLocaleString()}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
}
