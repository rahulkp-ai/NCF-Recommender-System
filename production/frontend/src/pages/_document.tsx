// apps/web/src/pages/_document.tsx
import { Html, Head, Main, NextScript } from "next/document";

export default function Document() {
  return (
    <Html lang="en">
      <Head>
        <meta charSet="utf-8" />
        <link rel="icon" href="/favicon.ico" />
        <meta name="theme-color" content="#141414" />
      </Head>
      <body className="bg-[#141414] text-white">
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
