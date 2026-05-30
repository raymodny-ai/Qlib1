'use client';

import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts';
import { useQuery } from '@tanstack/react-query';
import { getPortfolio } from '@/lib/api/client';
import { AppShell } from '@/components/layout/AppShell';

export default function PortfolioPage() {
  const { data: portfolio, isLoading } = useQuery({
    queryKey: ['portfolio', 'topk_dropout'],
    queryFn: () => getPortfolio('topk_dropout', { date: new Date().toISOString().split('T')[0] }),
  });

  const chartData = portfolio?.holdings.map((h) => ({
    name: h.instrument,
    value: h.weight * 100,
    score: h.score,
  })) || [];

  const COLORS = ['#0ea5e9', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16'];

  return (
    <AppShell>
      <Box>
        <Typography variant="h4" sx={{ mb: 3, fontWeight: 600 }}>
          Portfolio - {portfolio?.strategy_id || 'topk_dropout'}
        </Typography>

        <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
          {/* Pie Chart */}
          <Card sx={{ flex: '1 1 400px' }}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>
                Holdings Distribution
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={chartData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={2}
                    dataKey="value"
                    label={({ name, value }) => `${name}: ${value.toFixed(1)}%`}
                  >
                    {chartData.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value: number) => `${value.toFixed(2)}%`} />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Holdings Table */}
          <Card sx={{ flex: '1 1 400px' }}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>
                Top Holdings ({portfolio?.n_holdings || 0})
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Total Weight: {(portfolio?.total_weight || 0).toFixed(4)} | Cash: {((1 - (portfolio?.total_weight || 0)) * 100).toFixed(2)}%
              </Typography>
              <Box component="ul" sx={{ pl: 2, listStyle: 'none', p: 0, m: 0 }}>
                {portfolio?.holdings.map((holding) => (
                  <Box
                    key={holding.instrument}
                    component="li"
                    sx={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      py: 1,
                      borderBottom: '1px solid',
                      borderColor: 'divider',
                    }}
                  >
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                      {holding.instrument}
                    </Typography>
                    <Box sx={{ textAlign: 'right' }}>
                      <Typography variant="body2">{(holding.weight * 100).toFixed(2)}%</Typography>
                      {holding.score && (
                        <Typography variant="caption" color="text.secondary">
                          Score: {holding.score.toFixed(4)}
                        </Typography>
                      )}
                    </Box>
                  </Box>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Box>
      </Box>
    </AppShell>
  );
}