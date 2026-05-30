'use client';

import { useLocation, useNavigate } from 'react-router-dom';
import Box from '@mui/material/Box';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import Tooltip from '@mui/material/Tooltip';
import IconButton from '@mui/material/IconButton';
import Divider from '@mui/material/Divider';
import Typography from '@mui/material/Typography';
import DashboardIcon from '@mui/icons-material/Dashboard';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import AssessmentIcon from '@mui/icons-material/Assessment';
import HistoryIcon from '@mui/icons-material/History';
import SecurityIcon from '@mui/icons-material/Security';
import GavelIcon from '@mui/icons-material/Gavel';
import ArticleIcon from '@mui/icons-material/Article';
import StorageIcon from '@mui/icons-material/Storage';
import LockIcon from '@mui/icons-material/Lock';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import { useUIStore } from '@/store/uiStore';
import { useAuthStore, roleLabels } from '@/store/authStore';
import type { Permission } from '@/types/api';

interface MenuItem {
  label: string;
  icon: React.ReactNode;
  path: string;
  permission?: Permission;
  roles?: string[];
}

const menuItems: MenuItem[] = [
  { label: 'Dashboard', icon: <DashboardIcon />, path: '/' },
  { label: 'Data', icon: <StorageIcon />, path: '/data', permission: 'experiment:read' },
  { label: 'Factors', icon: <AssessmentIcon />, path: '/factors' },
  { label: 'Backtest', icon: <HistoryIcon />, path: '/backtest', permission: 'experiment:submit' },
  { label: 'PM Gate', icon: <GavelIcon />, path: '/gate', permission: 'signal:emergency_stop' },
  { label: 'Compliance', icon: <SecurityIcon />, path: '/compliance', permission: 'compliance:review' },
  { label: 'Logs', icon: <ArticleIcon />, path: '/logs', permission: 'logs:read' },
];

export function Sidebar() {
  const location = useLocation();
  const navigate = useNavigate();
  const pathname = location.pathname;
  const { sidebarOpen, sidebarCollapsed, toggleSidebar, setSidebarCollapsed } = useUIStore();
  const { role, hasPermission } = useAuthStore();

  const filteredMenuItems = menuItems.filter((item) => {
    if (item.roles) {
      return item.roles.includes(role);
    }
    if (item.permission) {
      return hasPermission(item.permission);
    }
    return true;
  });

  const sidebarWidth = sidebarCollapsed ? 64 : 240;

  return (
    <Box
      sx={{
        width: sidebarWidth,
        flexShrink: 0,
        position: 'fixed',
        left: 0,
        top: 0,
        bottom: 0,
        bgcolor: 'primary.main',
        color: 'primary.contrastText',
        display: 'flex',
        flexDirection: 'column',
        transition: 'width 0.3s',
        zIndex: 1200,
        overflow: 'hidden',
      }}
    >
      {/* Logo */}
      <Box
        sx={{
          height: 64,
          display: 'flex',
          alignItems: 'center',
          px: 2,
          borderBottom: '1px solid rgba(255,255,255,0.1)',
        }}
      >
        <Typography
          variant="h6"
          sx={{
            fontWeight: 700,
            whiteSpace: 'nowrap',
            opacity: sidebarCollapsed ? 0 : 1,
            transition: 'opacity 0.3s',
          }}
        >
          Qlib1
        </Typography>
      </Box>

      {/* Menu */}
      <List sx={{ flexGrow: 1, py: 1 }}>
        {filteredMenuItems.map((item) => {
          const isActive = pathname === item.path || (item.path !== '/' && pathname.startsWith(item.path));

          const menuItem = (
            <ListItemButton
              key={item.path}
              selected={isActive}
              onClick={() => navigate(item.path)}
              sx={{
                minHeight: 48,
                px: 2.5,
                bgcolor: isActive ? 'rgba(255,255,255,0.15)' : 'transparent',
                '&:hover': {
                  bgcolor: 'rgba(255,255,255,0.1)',
                },
              }}
            >
              <ListItemIcon
                sx={{
                  minWidth: 0,
                  mr: sidebarCollapsed ? 0 : 2,
                  justifyContent: 'center',
                  color: isActive ? 'inherit' : 'rgba(255,255,255,0.7)',
                }}
              >
                {item.icon}
              </ListItemIcon>
              <ListItemText
                primary={item.label}
                sx={{
                  opacity: sidebarCollapsed ? 0 : 1,
                  transition: 'opacity 0.3s',
                }}
              />
              {item.permission && !sidebarCollapsed && (
                <Tooltip title="Permission Required">
                  <LockIcon sx={{ fontSize: 14, color: 'rgba(255,255,255,0.4)', ml: 0.5 }} />
                </Tooltip>
              )}
            </ListItemButton>
          );

          return (
            <ListItem key={item.path} disablePadding sx={{ display: 'block' }}>
              {sidebarCollapsed ? (
                <Tooltip title={item.label} placement="right">
                  {menuItem}
                </Tooltip>
              ) : (
                menuItem
              )}
            </ListItem>
          );
        })}
      </List>

      {/* Role indicator */}
      <Box
        sx={{
          px: 2,
          py: 1.5,
          borderTop: '1px solid rgba(255,255,255,0.1)',
        }}
      >
        <Typography
          variant="caption"
          sx={{
            display: 'block',
            whiteSpace: 'nowrap',
            opacity: sidebarCollapsed ? 0 : 0.7,
          }}
        >
          {roleLabels[role]}
        </Typography>
      </Box>

      {/* Collapse toggle */}
      <Divider sx={{ borderColor: 'rgba(255,255,255,0.1)' }} />
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', p: 1 }}>
        <IconButton
          onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
          size="small"
          sx={{ color: 'rgba(255,255,255,0.7)' }}
        >
          {sidebarCollapsed ? <ChevronRightIcon /> : <ChevronLeftIcon />}
        </IconButton>
      </Box>
    </Box>
  );
}