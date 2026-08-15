// apps/web/src/hooks/useAuth.ts
import { useState, useEffect, useCallback } from "react";
import Cookies from "js-cookie";
import { authApi, User } from "../lib/api";

export function useAuth() {
  const [user, setUser]       = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const token = Cookies.get("ncf_token");
    if (!token) { setLoading(false); return; }
    authApi.profile()
      .then((res) => setUser(res.data))
      .catch(() => Cookies.remove("ncf_token"))
      .finally(() => setLoading(false));
  }, []);

  const login = useCallback(async (username: string, password: string) => {
    const res = await authApi.login({ username, password });
    Cookies.set("ncf_token", res.data.access_token, { expires: 1 });
    setUser(res.data.user);
    return res.data.user;
  }, []);

  const register = useCallback(async (
    username: string, email: string, password: string
  ) => {
    const res = await authApi.register({ username, email, password });
    Cookies.set("ncf_token", res.data.access_token, { expires: 1 });
    setUser(res.data.user);
    return res.data.user;
  }, []);

  const logout = useCallback(() => {
    Cookies.remove("ncf_token");
    setUser(null);
  }, []);

  return { user, loading, login, register, logout, isLoggedIn: !!user };
}
