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
import { useQuery } from '@tanstack/react-query';
import { getScores } from '@/lib/api/client';
import { AppShell } from '@/components/layout/AppShell';

export default function PredictionPage() {
  const { data: scores, isLoading } = useQuery({
    queryKey: ['scores'],
    queryFn: () => getScores({ limit: 50 }),
  });

  return (
    <AppShell>
      <Box>
        <Typography variant="h4" sx={{ mb: 3, fontWeight: 600 }}>
          Prediction Scores
        </Typography>

        <Card>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Rank</TableCell>
                  <TableCell>Instrument</TableCell>
                  <TableCell align="right">Score</TableCell>
                  <TableCell align="right">Percentile</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {isLoading ? (
                  Array.from({ length: 10 }).map((_, i) => (
                    <TableRow key={i}>
                      <TableCell><Skeleton width={40} /></TableCell>
                      <TableCell><Skeleton width={60} /></TableCell>
                      <TableCell><Skeleton width={80} /></TableCell>
                      <TableCell><Skeleton width={60} /></TableCell>
                    </TableRow>
                  ))
                ) : (
                  scores?.scores.map((item) => (
                    <TableRow key={item.instrument} hover>
                      <TableCell>#{item.rank}</TableCell>
                      <TableCell sx={{ fontWeight: 600 }}>{item.instrument}</TableCell>
                      <TableCell
                        align="right"
                        sx={{
                          color: item.score > 0 ? 'success.main' : 'error.main',
                          fontWeight: 600,
                        }}
                      >
                        {item.score.toFixed(6)}
                      </TableCell>
                      <TableCell align="right">
                        <Chip
                          label={`${item.percentile.toFixed(1)}%`}
                          size="small"
                          color={item.percentile > 90 ? 'success' : item.percentile > 50 ? 'primary' : 'default'}
                        />
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