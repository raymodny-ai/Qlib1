'use client';

import { useState } from 'react';
import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import { useMutation } from '@tanstack/react-query';
import { queryFactors } from '@/lib/api/client';
import { AppShell } from '@/components/layout/AppShell';

export default function FactorsPage() {
  const [instruments, setInstruments] = useState('AAPL,MSFT,GOOGL');
  const [startDate, setStartDate] = useState('2025-01-01');
  const [endDate, setEndDate] = useState('2025-12-31');
  const [dataset, setDataset] = useState('fundamentals');

  const mutation = useMutation({
    mutationFn: () =>
      queryFactors(dataset, {
        instruments: instruments.split(',').map((s) => s.trim()),
        start_date: startDate,
        end_date: endDate,
      }),
  });

  return (
    <AppShell>
      <Box>
        <Typography variant="h4" sx={{ mb: 3, fontWeight: 600 }}>
          Factor Data
        </Typography>

        {/* Query Form */}
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'flex-end' }}>
              <TextField
                label="Instruments"
                value={instruments}
                onChange={(e) => setInstruments(e.target.value)}
                size="small"
                sx={{ minWidth: 250 }}
                helperText="Comma-separated symbols"
              />
              <TextField
                label="Start Date"
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                size="small"
                InputLabelProps={{ shrink: true }}
              />
              <TextField
                label="End Date"
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                size="small"
                InputLabelProps={{ shrink: true }}
              />
              <TextField
                label="Dataset"
                value={dataset}
                onChange={(e) => setDataset(e.target.value)}
                size="small"
                sx={{ minWidth: 150 }}
              />
              <Button
                variant="contained"
                onClick={() => mutation.mutate()}
                disabled={mutation.isPending}
              >
                {mutation.isPending ? <CircularProgress size={24} /> : 'Query'}
              </Button>
            </Box>
          </CardContent>
        </Card>

        {/* Results */}
        {mutation.isError && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {mutation.error.message || 'Failed to query factors'}
          </Alert>
        )}

        {mutation.data && (
          <Card>
            <CardContent>
              <Typography variant="subtitle2" sx={{ mb: 2 }}>
                Results: {mutation.data.n_rows} rows, {mutation.data.n_fields} fields
              </Typography>
              <TableContainer sx={{ maxHeight: 500 }}>
                <Table stickyHeader size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Instrument</TableCell>
                      <TableCell>Date</TableCell>
                      {mutation.data.data[0] &&
                        Object.keys(mutation.data.data[0])
                          .filter((k) => k !== 'instrument' && k !== 'date')
                          .map((field) => (
                            <TableCell key={field}>{field}</TableCell>
                          ))}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {mutation.data.data.slice(0, 100).map((row, i) => (
                      <TableRow key={i}>
                        <TableCell sx={{ fontWeight: 600 }}>{row.instrument}</TableCell>
                        <TableCell>{row.date}</TableCell>
                        {Object.entries(row)
                          .filter(([k]) => k !== 'instrument' && k !== 'date')
                          .map(([key, value]) => (
                            <TableCell key={key}>
                              {typeof value === 'number' ? value.toFixed(4) : value}
                            </TableCell>
                          ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        )}
      </Box>
    </AppShell>
  );
}