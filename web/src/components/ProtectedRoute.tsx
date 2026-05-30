import { Navigate } from 'react-router-dom';
import { useAuthStore } from '@/store/authStore';
import type { Permission } from '@/types/api';

interface ProtectedRouteProps {
  children: React.ReactNode;
  permission?: Permission;
}

export function ProtectedRoute({ children, permission }: ProtectedRouteProps) {
  const { isAuthenticated, hasPermission, role } = useAuthStore();

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  if (permission && !hasPermission(permission)) {
    return <Navigate to="/access-denied" replace state={{ required: permission, role }} />;
  }

  return <>{children}</>;
}
