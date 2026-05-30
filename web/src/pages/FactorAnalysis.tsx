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
import ToggleButton from '@mui/material/ToggleButton';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';
import MenuItem from '@mui/material/MenuItem';
import Chip from '@mui/material/Chip';
import Autocomplete from '@mui/material/Autocomplete';
import DownloadIcon from '@mui/icons-material/Download';
import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { queryFactors } from '@/lib/api/client';
import { FactorTimeSeries, generateMockFactorTimeSeries } from '@/components/charts/FactorTimeSeries';
import { FactorHistogram, generateMockHistogram } from '@/components/charts/FactorHistogram';
import { ICSummaryCard, generateMockICSummary } from '@/components/factors/ICSummaryCard';
import type { FactorResponse } from '@/types/api';

type ViewMode = 'table' | 'timeseries' | 'histogram';

const datasetOptions = ['alpha158', 'alpha360', 'alpha101'];

export function FactorAnalysisPage() {
  const [dataset, setDataset] = useState('alpha158');
  const [instruments, setInstruments] = useState<string[]>(['AAPL', 'MSFT', 'GOOGL']);
  const [startDate, setStartDate] = useState('2023-01-01');
  const [endDate, setEndDate] = useState('2024-01-01');
  const [viewMode, setViewMode] = useState<ViewMode>('table');
  const [selectedInstrument, setSelectedInstrument] = useState('');
  const [selectedFactor, setSelectedFactor] = useState('');

  const { data, isLoading, refetch } = useQuery({
    queryKey: ['factors', dataset, instruments.join(','), startDate, endDate],
    queryFn: () =>
      queryFactors(dataset, {
        instruments,
        start_date: startDate,
        end_date: endDate,
      }),
    enabled: false,
  });

  const factorNames = useMemo(() => {
    if (!data || data.data.length === 0) return [];
    return Object.keys(data.data[0]).filter((k) => !['instrument', 'date'].includes(k));
  }, [data]);

  // Mock chart data
  const mockIC = useMemo(() => generateMockICSummary(), []);
  const mockDates = useMemo(() => {
    const dates: string[] = [];
    const start = new Date(startDate);
    const end = new Date(endDate);
    const cur = new Date(start);
    while (cur <= end) {
      dates.push(cur.toISOString().split('T')[0]);
      cur.setDate(cur.getDate() + 1);
    }
    return dates;
  }, [startDate, endDate]);
  const mockTimeSeries = useMemo(
    () => generateMockFactorTimeSeries(factorNames.slice(0, 5), mockDates),
    [factorNames, mockDates]
  );
  const mockHistogram = useMemo(
    () => generateMockHistogram(selectedFactor || factorNames[0] || 'factor'),
    [selectedFactor, factorNames]
  );

  // CSV export
  const handleExportCSV = () => {
    if (!data || data.data.length === 0) return;
    const rows = data.data;
    const headers = ['instrument', 'date', ...factorNames];
    const csvLines = [
      headers.join(','),
      ...rows.map((row) => headers.map((h) => String(row[h] ?? '')).join(',')),
    ];
    const csv = csvLines.join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `factors_${dataset}_${startDate}_${endDate}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

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
                select
                label="Dataset"
                value={dataset}
                onChange={(e) => setDataset(e.target.value)}
                size="small"
              >
                {datasetOptions.map((opt) => (
                  <MenuItem key={opt} value={opt}>{opt}</MenuItem>
                ))}
              </TextField>
            </Grid>
            <Grid item xs={12} md={3}>
              <Autocomplete
                multiple
                freeSolo
                options={[]}
                value={instruments}
                onChange={(_, val) => setInstruments(val)}
                renderTags={(value, getTagProps) =>
                  value.map((option, index) => (
                    <Chip
                      label={option}
                      size="small"
                      {...getTagProps({ index })}
                      key={index}
                    />
                  ))
                }
                renderInput={(params) => (
                  <TextField
                    {...params}
                    label="Instruments"
                    size="small"
                    placeholder="AAPL, MSFT..."
                  />
                )}
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
        <>
          {/* View controls */}
          <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 1 }}>
            <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
              <ToggleButtonGroup
                value={viewMode}
                exclusive
                onChange={(_, val) => val && setViewMode(val)}
                size="small"
              >
                <ToggleButton value="table">Table</ToggleButton>
                <ToggleButton value="timeseries">Time Series</ToggleButton>
                <ToggleButton value="histogram">Histogram</ToggleButton>
              </ToggleButtonGroup>
              {viewMode === 'timeseries' && (
                <TextField
                  select
                  label="Instrument"
                  value={selectedInstrument || data.data[0]?.instrument || ''}
                  onChange={(e) => setSelectedInstrument(e.target.value)}
                  size="small"
                  sx={{ minWidth: 120 }}
                >
                  {[...new Set(data.data.map((r) => r.instrument))].map((inst) => (
                    <MenuItem key={inst} value={inst}>{inst}</MenuItem>
                  ))}
                </TextField>
              )}
              {(viewMode === 'timeseries' || viewMode === 'histogram') && (
                <TextField
                  select
                  label="Factor"
                  value={selectedFactor || factorNames[0] || ''}
                  onChange={(e) => setSelectedFactor(e.target.value)}
                  size="small"
                  sx={{ minWidth: 140 }}
                >
                  {factorNames.map((f) => (
                    <MenuItem key={f} value={f}>{f}</MenuItem>
                  ))}
                </TextField>
              )}
            </Box>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button
                variant="outlined"
                size="small"
                startIcon={<DownloadIcon />}
                onClick={handleExportCSV}
              >
                Export CSV
              </Button>
            </Box>
          </Box>

          {/* IC Summary */}
          <Box sx={{ mb: 3 }}>
            <ICSummaryCard data={mockIC} />
          </Box>

          {/* Chart views */}
          {viewMode === 'timeseries' && (
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <FactorTimeSeries
                  data={mockTimeSeries}
                  factors={factorNames.slice(0, 5)}
                  selectedFactor={selectedFactor || undefined}
                />
              </CardContent>
            </Card>
          )}

          {viewMode === 'histogram' && (
            <Card sx={{ mb: 3 }}>
              <CardContent>
                <FactorHistogram
                  data={mockHistogram}
                  factorName={selectedFactor || factorNames[0] || 'factor'}
                />
              </CardContent>
            </Card>
          )}

          {/* Table view */}
          {viewMode === 'table' && (
            <>
              <TableContainer component={Paper}>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Instrument</TableCell>
                      <TableCell>Date</TableCell>
                      {factorNames.map((field) => (
                        <TableCell key={field}>{field}</TableCell>
                      ))}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {data.data.slice(0, 50).map((row, idx) => (
                      <TableRow key={idx}>
                        <TableCell>{row.instrument}</TableCell>
                        <TableCell>{row.date}</TableCell>
                        {factorNames.map((field) => (
                          <TableCell key={field}>
                            {typeof row[field] === 'number' ? (row[field] as number).toFixed(4) : String(row[field])}
                          </TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                Showing first 50 of {data.data.length} rows
              </Typography>
            </>
          )}
        </>
      )}
    </Box>
  );
}
