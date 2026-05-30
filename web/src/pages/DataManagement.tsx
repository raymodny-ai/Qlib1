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
import { useQuery } from '@tanstack/react-query';
import { getInstruments } from '@/lib/api/client';

export function DataManagementPage() {
  const { data: instruments, isLoading } = useQuery({
    queryKey: ['instruments'],
    queryFn: () => getInstruments({ limit: 50 }),
  });

  return (
    <Box>
      <Typography variant="h4" sx={{ mb: 3, fontWeight: 600 }}>
        Data Management
      </Typography>

      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 1 }}>
                Instrument Count
              </Typography>
              <Typography variant="h3" sx={{ fontWeight: 700, color: 'primary.main' }}>
                {instruments?.length ?? '-'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 1 }}>
                Data Sources
              </Typography>
              <Chip label="Alpha158" color="primary" size="small" sx={{ mr: 1 }} />
              <Chip label="Alpha360" color="secondary" size="small" />
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>
                Ingestion
              </Typography>
              <Button variant="contained" fullWidth>
                Trigger Data Ingestion
              </Button>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Typography variant="h5" sx={{ mb: 2 }}>
        Instruments
      </Typography>
      <TableContainer component={Paper}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Symbol</TableCell>
              <TableCell>Name</TableCell>
              <TableCell>Sector</TableCell>
              <TableCell>Industry</TableCell>
              <TableCell>Market Cap</TableCell>
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
                <TableCell>{inst.market_cap ? `$${(inst.market_cap / 1e9).toFixed(1)}B` : '-'}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
}
