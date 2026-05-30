'use client';

import { useState } from 'react';
import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import TextField from '@mui/material/TextField';
import Autocomplete from '@mui/material/Autocomplete';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Skeleton from '@mui/material/Skeleton';
import { useQuery } from '@tanstack/react-query';
import { getInstruments } from '@/lib/api/client';
import { AppShell } from '@/components/layout/AppShell';
import type { InstrumentInfo } from '@/types/api';

export default function InstrumentsPage() {
  const [sectorFilter, setSectorFilter] = useState<string | null>(null);
  const [selectedInstruments, setSelectedInstruments] = useState<InstrumentInfo[]>([]);

  const { data: instruments, isLoading } = useQuery({
    queryKey: ['instruments', sectorFilter],
    queryFn: () => getInstruments({ sector: sectorFilter || undefined, limit: 100 }),
  });

  const sectors = [...new Set(instruments?.map((i) => i.sector).filter(Boolean))] as string[];

  return (
    <AppShell>
      <Box>
        <Typography variant="h4" sx={{ mb: 3, fontWeight: 600 }}>
          Instruments
        </Typography>

        {/* Filters */}
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
              <Autocomplete
                options={sectors}
                value={sectorFilter}
                onChange={(_, value) => setSectorFilter(value)}
                renderInput={(params) => (
                  <TextField {...params} label="Sector Filter" size="small" sx={{ minWidth: 200 }} />
                )}
              />
              <Autocomplete
                multiple
                options={instruments || []}
                getOptionLabel={(option) => `${option.symbol} - ${option.name || option.symbol}`}
                value={selectedInstruments}
                onChange={(_, value) => setSelectedInstruments(value)}
                renderInput={(params) => (
                  <TextField {...params} label="Search Instruments" size="small" sx={{ minWidth: 300 }} />
                )}
                sx={{ minWidth: 400 }}
              />
            </Box>
          </CardContent>
        </Card>

        {/* Table */}
        <Card>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Symbol</TableCell>
                  <TableCell>Name</TableCell>
                  <TableCell>Sector</TableCell>
                  <TableCell>Market Cap</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {isLoading ? (
                  Array.from({ length: 10 }).map((_, i) => (
                    <TableRow key={i}>
                      <TableCell><Skeleton width={60} /></TableCell>
                      <TableCell><Skeleton width={150} /></TableCell>
                      <TableCell><Skeleton width={100} /></TableCell>
                      <TableCell><Skeleton width={80} /></TableCell>
                    </TableRow>
                  ))
                ) : (
                  instruments?.map((instrument) => (
                    <TableRow key={instrument.symbol} hover>
                      <TableCell sx={{ fontWeight: 600 }}>{instrument.symbol}</TableCell>
                      <TableCell>{instrument.name || '-'}</TableCell>
                      <TableCell>
                        {instrument.sector && (
                          <Typography variant="caption" sx={{ bgcolor: 'primary.light', px: 1, py: 0.5, borderRadius: 1 }}>
                            {instrument.sector}
                          </Typography>
                        )}
                      </TableCell>
                      <TableCell>
                        {instrument.market_cap
                          ? `$${(instrument.market_cap / 1e12).toFixed(2)}T`
                          : '-'}
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