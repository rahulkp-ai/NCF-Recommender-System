import Head from "next/head";

export default function DashboardPage() {
  return (
    <>
      <Head>
        <title>NCF Benchmark Dashboard</title>
      </Head>
      <iframe
        src="/dashboard/index.html"
        title="NCF Benchmark Dashboard"
        className="h-screen w-full border-0 bg-[#0a0a0a]"
      />
    </>
  );
}
