import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { SnackbarProvider } from 'notistack';
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';
import { useState, useMemo, useEffect } from 'react';
import { AppShell } from '@/components/layout/AppShell';
import { ProtectedRoute } from '@/components/ProtectedRoute';
import { ErrorBoundary } from '@/components/ErrorBoundary';
import { DashboardPage } from '@/pages/Dashboard';
import { LoginPage } from '@/pages/Login';
import { FactorAnalysisPage } from '@/pages/FactorAnalysis';
import { BacktestPage } from '@/pages/Backtest';
import { PMGatePage } from '@/pages/PMGate';
import { CompliancePage } from '@/pages/Compliance';
import { DataManagementPage } from '@/pages/DataManagement';
import { AccessDeniedPage } from '@/pages/AccessDenied';
import { useUIStore } from '@/store/uiStore';

const typography = {
  fontFamily: 'Inter, system-ui, sans-serif',
  h1: { fontSize: '2rem', fontWeight: 600 },
  h2: { fontSize: '1.5rem', fontWeight: 600 },
  h3: { fontSize: '1.25rem', fontWeight: 600 },
  body1: { fontSize: '0.875rem' },
  body2: { fontSize: '0.75rem' },
} as const;

const components = {
  MuiButton: {
    styleOverrides: {
      root: { textTransform: 'none' as const },
    },
  },
  MuiCard: {
    styleOverrides: {
      root: { boxShadow: '0 1px 3px 0 rgb(0 0 0 / 0.1)' },
    },
  },
} as const;

function createAppTheme(mode: 'light' | 'dark') {
  return createTheme({
    palette: {
      mode,
      primary: {
        main: mode === 'light' ? '#0ea5e9' : '#38bdf8',
        light: '#38bdf8',
        dark: '#0284c7',
      },
      secondary: {
        main: '#64748b',
      },
      success: { main: '#22c55e' },
      warning: { main: '#f59e0b' },
      error: { main: mode === 'light' ? '#ef4444' : '#f87171' },
      background: mode === 'light'
        ? { default: '#f8fafc', paper: '#ffffff' }
        : { default: '#020617', paper: '#1e293b' },
    },
    typography,
    components,
  });
}

function useThemeMode(): 'light' | 'dark' {
  const themeSetting = useUIStore((s) => s.theme);
  const [systemPrefersDark, setSystemPrefersDark] = useState(
    () => window.matchMedia('(prefers-color-scheme: dark)').matches
  );

  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const handler = (e: MediaQueryListEvent) => setSystemPrefersDark(e.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);

  if (themeSetting === 'system') {
    return systemPrefersDark ? 'dark' : 'light';
  }
  return themeSetting === 'dark' ? 'dark' : 'light';
}

export function App() {
  const mode = useThemeMode();
  const theme = useMemo(() => createAppTheme(mode), [mode]);

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
              <Route path="/login" element={<ErrorBoundary><LoginPage /></ErrorBoundary>} />
              <Route
                path="/"
                element={
                  <ProtectedRoute>
                    <AppShell>
                      <ErrorBoundary><DashboardPage /></ErrorBoundary>
                    </AppShell>
                  </ProtectedRoute>
                }
              />
              <Route
                path="/factors"
                element={
                  <ProtectedRoute>
                    <AppShell>
                      <ErrorBoundary><FactorAnalysisPage /></ErrorBoundary>
                    </AppShell>
                  </ProtectedRoute>
                }
              />
              <Route
                path="/backtest"
                element={
                  <ProtectedRoute>
                    <AppShell>
                      <ErrorBoundary><BacktestPage /></ErrorBoundary>
                    </AppShell>
                  </ProtectedRoute>
                }
              />
              <Route
                path="/gate"
                element={
                  <ProtectedRoute>
                    <AppShell>
                      <ErrorBoundary><PMGatePage /></ErrorBoundary>
                    </AppShell>
                  </ProtectedRoute>
                }
              />
              <Route
                path="/compliance"
                element={
                  <ProtectedRoute permission="compliance:review">
                    <AppShell>
                      <ErrorBoundary><CompliancePage /></ErrorBoundary>
                    </AppShell>
                  </ProtectedRoute>
                }
              />
              <Route
                path="/data"
                element={
                  <ProtectedRoute permission="experiment:read">
                    <AppShell>
                      <ErrorBoundary><DataManagementPage /></ErrorBoundary>
                    </AppShell>
                  </ProtectedRoute>
                }
              />
              <Route
                path="/access-denied"
                element={
                  <AppShell>
                    <ErrorBoundary><AccessDeniedPage /></ErrorBoundary>
                  </AppShell>
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
