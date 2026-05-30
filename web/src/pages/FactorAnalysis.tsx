import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import Grid from '@mui/material/Grid';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { queryFactors } from '@/lib/api/client';

export function FactorAnalysisPage() {
  const [dataset, setDataset] = useState('alpha158');
  const [instruments, setInstruments] = useState('AAPL,MSFT,GOOGL');
  const [startDate, setStartDate] = useState('2023-01-01');
  const [endDate, setEndDate] = useState('2024-01-01');

  const { data, isLoading, refetch } = useQuery({
    queryKey: ['factors', dataset, instruments, startDate, endDate],
    queryFn: () =>
      queryFactors(dataset, {
        instruments: instruments.split(',').map((s) => s.trim()),
        start_date: startDate,
        end_date: endDate,
      }),
    enabled: false,
  });

  return (
    <Box>
      <Typography variant="h4" sx={{ mb: 3, fontWeight: 600 }}>
        Factor Analysis
      </Typography>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={3}>
              <TextField
                fullWidth
                label="Dataset"
                value={dataset}
                onChange={(e) => setDataset(e.target.value)}
                size="small"
              />
            </Grid>
            <Grid item xs={12} md={3}>
              <TextField
                fullWidth
                label="Instruments (comma-separated)"
                value={instruments}
                onChange={(e) => setInstruments(e.target.value)}
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
            <Grid item xs={12} md={2}>
              <Button variant="contained" fullWidth onClick={() => refetch()} disabled={isLoading}>
                {isLoading ? 'Loading...' : 'Query Factors'}
              </Button>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {data && (
        <TableContainer component={Paper}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Instrument</TableCell>
                <TableCell>Date</TableCell>
                {data.data.length > 0 &&
                  Object.keys(data.data[0])
                    .filter((k) => !['instrument', 'date'].includes(k))
                    .map((field) => <TableCell key={field}>{field}</TableCell>)}
              </TableRow>
            </TableHead>
            <TableBody>
              {data.data.slice(0, 50).map((row, idx) => (
                <TableRow key={idx}>
                  <TableCell>{row.instrument}</TableCell>
                  <TableCell>{row.date}</TableCell>
                  {Object.keys(row)
                    .filter((k) => !['instrument', 'date'].includes(k))
                    .map((field) => (
                      <TableCell key={field}>
                        {typeof row[field] === 'number' ? (row[field] as number).toFixed(4) : String(row[field])}
                      </TableCell>
                    ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Box>
  );
}
