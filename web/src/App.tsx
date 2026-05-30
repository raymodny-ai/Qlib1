import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { SnackbarProvider } from 'notistack';
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';
import { useState } from 'react';
import { AppShell } from '@/components/layout/AppShell';
import { ProtectedRoute } from '@/components/ProtectedRoute';
import { DashboardPage } from '@/pages/Dashboard';
import { LoginPage } from '@/pages/Login';
import { FactorAnalysisPage } from '@/pages/FactorAnalysis';
import { BacktestPage } from '@/pages/Backtest';
import { PMGatePage } from '@/pages/PMGate';
import { CompliancePage } from '@/pages/Compliance';
import { DataManagementPage } from '@/pages/DataManagement';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#0ea5e9',
      light: '#38bdf8',
      dark: '#0284c7',
    },
    secondary: {
      main: '#64748b',
    },
    success: {
      main: '#22c55e',
    },
    warning: {
      main: '#f59e0b',
    },
    error: {
      main: '#ef4444',
    },
    background: {
      default: '#f8fafc',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: 'Inter, system-ui, sans-serif',
    h1: { fontSize: '2rem', fontWeight: 600 },
    h2: { fontSize: '1.5rem', fontWeight: 600 },
    h3: { fontSize: '1.25rem', fontWeight: 600 },
    body1: { fontSize: '0.875rem' },
    body2: { fontSize: '0.75rem' },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: { textTransform: 'none' },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: { boxShadow: '0 1px 3px 0 rgb(0 0 0 / 0.1)' },
      },
    },
  },
});

export function App() {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 60 * 1000,
            refetchOnWindowFocus: false,
          },
        },
      })
  );

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <SnackbarProvider maxSnack={3} autoHideDuration={3000}>
          <BrowserRouter>
            <Routes>
              <Route path="/login" element={<LoginPage />} />
              <Route
                path="/"
                element={
                  <ProtectedRoute>
                    <AppShell>
                      <DashboardPage />
                    </AppShell>
                  </ProtectedRoute>
                }
              />
              <Route
                path="/factors"
                element={
                  <ProtectedRoute>
                    <AppShell>
                      <FactorAnalysisPage />
                    </AppShell>
                  </ProtectedRoute>
                }
              />
              <Route
                path="/backtest"
                element={
                  <ProtectedRoute>
                    <AppShell>
                      <BacktestPage />
                    </AppShell>
                  </ProtectedRoute>
                }
              />
              <Route
                path="/gate"
                element={
                  <ProtectedRoute>
                    <AppShell>
                      <PMGatePage />
                    </AppShell>
                  </ProtectedRoute>
                }
              />
              <Route
                path="/compliance"
                element={
                  <ProtectedRoute>
                    <AppShell>
                      <CompliancePage />
                    </AppShell>
                  </ProtectedRoute>
                }
              />
              <Route
                path="/data"
                element={
                  <ProtectedRoute>
                    <AppShell>
                      <DataManagementPage />
                    </AppShell>
                  </ProtectedRoute>
                }
              />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </BrowserRouter>
        </SnackbarProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}
