// apps/web/src/pages/_app.tsx
import type { AppProps } from "next/app";
import Navbar from "../components/layout/Navbar";
import "../styles/globals.css";

export default function App({ Component, pageProps }: AppProps) {
  return (
    <>
      <Navbar />
      <Component {...pageProps} />
    </>
  );
}
