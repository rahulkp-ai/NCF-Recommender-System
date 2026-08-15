// Placeholder test to establish the test setup (Jest + React Testing
// Library). Replace with real component tests during Phase 5.
import { render, screen } from "@testing-library/react";

describe("test setup", () => {
  it("runs", () => {
    render(<div>NCF Recommender</div>);
    expect(screen.getByText("NCF Recommender")).toBeInTheDocument();
  });
});
