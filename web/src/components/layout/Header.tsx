'use client';

import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import Tooltip from '@mui/material/Tooltip';
import Badge from '@mui/material/Badge';
import Chip from '@mui/material/Chip';
import MenuIcon from '@mui/icons-material/Menu';
import NotificationsIcon from '@mui/icons-material/Notifications';
import PersonIcon from '@mui/icons-material/Person';
import Menu from '@mui/material/Menu';
import MenuItem from '@mui/material/MenuItem';
import ListItemIcon from '@mui/material/ListItemIcon';
import Divider from '@mui/material/Divider';
import Select from '@mui/material/Select';
import FormControl from '@mui/material/FormControl';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';
import { useState } from 'react';
import { useUIStore } from '@/store/uiStore';
import { useAuthStore, roleLabels } from '@/store/authStore';
import { useGateStore } from '@/store/gateStore';

export function Header() {
  const { toggleSidebar, health, healthError, theme, setTheme } = useUIStore();
  const { userId, role, logout } = useAuthStore();
  const { status: gateStatus } = useGateStore();

  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [userMenuAnchor, setUserMenuAnchor] = useState<null | HTMLElement>(null);

  const handleNotificationClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleUserMenuClick = (event: React.MouseEvent<HTMLElement>) => {
    setUserMenuAnchor(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
    setUserMenuAnchor(null);
  };

  const isHealthy = health?.status === 'healthy';
  const isGateClosed = gateStatus?.is_any_closed ?? false;

  const isDark = theme === 'dark' || (theme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);

  return (
    <Box
      component="header"
      sx={{
        height: 64,
        display: 'flex',
        alignItems: 'center',
        px: 3,
        bgcolor: 'background.paper',
        borderBottom: '1px solid',
        borderColor: 'divider',
        gap: 2,
      }}
    >
      {/* Menu toggle */}
      <IconButton onClick={toggleSidebar} edge="start">
        <MenuIcon />
      </IconButton>

      {/* Gate status alert */}
      {isGateClosed && (
        <Chip
          label="Gate Closed"
          color="error"
          size="small"
          sx={{ fontWeight: 600 }}
        />
      )}

      {/* Spacer */}
      <Box sx={{ flexGrow: 1 }} />

      {/* Theme toggle */}
      <Tooltip title={isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}>
        <IconButton onClick={() => setTheme(isDark ? 'light' : 'dark')}>
          {isDark ? <Brightness7Icon /> : <Brightness4Icon />}
        </IconButton>
      </Tooltip>

      {/* User role selector (dev mode only) */}
      {import.meta.env.DEV && (
        <FormControl size="small" sx={{ minWidth: 150 }}>
          <Select
            value={userId}
            onChange={(e) => {
              useAuthStore.getState().login(e.target.value);
            }}
            displayEmpty
            startAdornment={
              <ListItemIcon sx={{ minWidth: 32 }}>
                <PersonIcon fontSize="small" />
              </ListItemIcon>
            }
          >
            <MenuItem value="admin">Admin</MenuItem>
            <MenuItem value="researcher">Researcher</MenuItem>
            <MenuItem value="pm">Portfolio Manager</MenuItem>
            <MenuItem value="auditor">Auditor</MenuItem>
          </Select>
        </FormControl>
      )}

      {/* Notifications */}
      <IconButton onClick={handleNotificationClick}>
        <Badge badgeContent={0} color="error">
          <NotificationsIcon />
        </Badge>
      </IconButton>

      {/* Health status */}
      <Chip
        label={isHealthy ? 'Online' : 'Offline'}
        color={isHealthy ? 'success' : 'error'}
        size="small"
        variant="outlined"
      />

      {/* User menu */}
      <IconButton onClick={handleUserMenuClick}>
        <PersonIcon />
      </IconButton>

      {/* Notification menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
      >
        <MenuItem onClick={handleClose}>
          <Typography variant="body2">No new notifications</Typography>
        </MenuItem>
      </Menu>

      {/* User menu */}
      <Menu
        anchorEl={userMenuAnchor}
        open={Boolean(userMenuAnchor)}
        onClose={handleClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
      >
        <Box sx={{ px: 2, py: 1 }}>
          <Typography variant="subtitle2">{roleLabels[role]}</Typography>
          <Typography variant="caption" color="text.secondary">
            {userId}
          </Typography>
        </Box>
        <Divider />
        <MenuItem onClick={() => { logout(); handleClose(); }}>Logout</MenuItem>
      </Menu>
    </Box>
  );
}