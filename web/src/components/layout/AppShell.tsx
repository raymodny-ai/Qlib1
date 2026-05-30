'use client';

import { useEffect } from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import { useQuery } from '@tanstack/react-query';
import { getHealth } from '@/lib/api/client';
import { useUIStore } from '@/store/uiStore';
import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';

interface AppShellProps {
  children: React.ReactNode;
}

export function AppShell({ children }: AppShellProps) {
  const { sidebarOpen } = useUIStore();

  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: getHealth,
    refetchInterval: 30000, // 30 seconds polling
    staleTime: 10000,
  });

  useEffect(() => {
    if (health) {
      useUIStore.getState().setHealth(health);
    }
  }, [health]);

  return (
    <Box
      sx={{
        display: 'flex',
        minHeight: '100vh',
        bgcolor: 'background.default',
      }}
    >
      <Sidebar />
      <Box
        sx={{
          flexGrow: 1,
          display: 'flex',
          flexDirection: 'column',
          ml: sidebarOpen ? 28 : 8,
          transition: 'margin-left 0.3s',
        }}
      >
        <Header />
        <Box
          component="main"
          sx={{
            flexGrow: 1,
            p: 3,
            bgcolor: 'background.default',
          }}
        >
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            {health?.status === 'healthy' ? 'System Online' : 'System Issues Detected'}
          </Typography>
          {children}
        </Box>
      </Box>
    </Box>
  );
}