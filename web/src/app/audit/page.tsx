'use client';

import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Chip from '@mui/material/Chip';
import Skeleton from '@mui/material/Skeleton';
import Alert from '@mui/material/Alert';
import { useQuery } from '@tanstack/react-query';
import { getAuditLogs, verifyAuditChain } from '@/lib/api/client';
import { AppShell } from '@/components/layout/AppShell';

export default function AuditPage() {
  const { data: logs, isLoading: logsLoading } = useQuery({
    queryKey: ['audit-logs'],
    queryFn: () => getAuditLogs({ limit: 50 }),
  });

  const { data: chainVerification, isLoading: chainLoading } = useQuery({
    queryKey: ['audit-chain'],
    queryFn: () => verifyAuditChain(),
  });

  return (
    <AppShell>
      <Box>
        <Typography variant="h4" sx={{ mb: 3, fontWeight: 600 }}>
          Audit Logs
        </Typography>

        {/* Chain Verification */}
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 2 }}>
              Audit Chain Integrity
            </Typography>
            {chainLoading ? (
              <Skeleton width={200} />
            ) : chainVerification ? (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Chip
                  label={chainVerification.verified ? 'VERIFIED' : 'BROKEN'}
                  color={chainVerification.verified ? 'success' : 'error'}
                  sx={{ fontWeight: 700 }}
                />
                <Typography variant="body2">
                  {chainVerification.total_entries} entries | Date: {chainVerification.date}
                </Typography>
                {!chainVerification.verified && chainVerification.broken_at && (
                  <Alert severity="error" sx={{ ml: 2 }}>
                    Chain broken at entry #{chainVerification.broken_at}
                  </Alert>
                )}
              </Box>
            ) : null}
          </CardContent>
        </Card>

        {/* Logs Table */}
        <Card>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Timestamp</TableCell>
                  <TableCell>Event</TableCell>
                  <TableCell>User</TableCell>
                  <TableCell>Details</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {logsLoading ? (
                  Array.from({ length: 5 }).map((_, i) => (
                    <TableRow key={i}>
                      <TableCell><Skeleton width={150} /></TableCell>
                      <TableCell><Skeleton width={100} /></TableCell>
                      <TableCell><Skeleton width={80} /></TableCell>
                      <TableCell><Skeleton width={200} /></TableCell>
                    </TableRow>
                  ))
                ) : logs?.entries.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={4} sx={{ textAlign: 'center', py: 4 }}>
                      <Typography variant="body2" color="text.secondary">
                        No audit logs found
                      </Typography>
                    </TableCell>
                  </TableRow>
                ) : (
                  logs?.entries.map((entry) => (
                    <TableRow key={entry.event_id} hover>
                      <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>
                        {new Date(entry.timestamp).toLocaleString()}
                      </TableCell>
                      <TableCell>
                        <Chip label={entry.event_type} size="small" variant="outlined" />
                      </TableCell>
                      <TableCell>{entry.user_id}</TableCell>
                      <TableCell sx={{ fontSize: '0.75rem', maxWidth: 400 }}>
                        {JSON.stringify(entry.details).slice(0, 100)}
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </Card>
      </Box>
    </AppShell>
  );
}