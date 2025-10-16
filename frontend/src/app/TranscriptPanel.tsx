import { useEffect, useRef } from "react";
import clsx from "clsx";
import { ChatMessage } from "./chatHistory";

interface TranscriptPanelProps {
  chatHistory: ChatMessage[];
  isVisible: boolean;
}

const TranscriptPanel = ({ chatHistory, isVisible }: TranscriptPanelProps) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (isVisible && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [chatHistory, isVisible]);

  const getRoleLabel = (role: string) => {
    return role === "user" ? "You" : "Assistant";
  };

  const getRoleColor = (role: string) => {
    return role === "user" ? "text-blue-400" : "text-green";
  };

  if (!isVisible) {
    return null;
  }

  return (
    <div
      ref={containerRef}
      className="fixed bottom-0 left-0 right-0 bg-background border-t-2 border-lightgray z-20 max-h-96 overflow-y-auto"
    >
      <div className="max-w-4xl mx-auto px-4 py-6">
        <div className="flex flex-col gap-4">
          {chatHistory.length === 0 ? (
            <p className="text-lightgray text-center italic">
              No conversation yet. Start talking to see the transcript here.
            </p>
          ) : (
            chatHistory.map((message, index) => (
              <div
                key={`${message.role}-${index}`}
                className="flex flex-col gap-1"
              >
                <span
                  className={clsx(
                    "font-semibold text-sm",
                    getRoleColor(message.role)
                  )}
                >
                  {getRoleLabel(message.role)}
                </span>
                <p className="text-white whitespace-pre-wrap">
                  {message.content}
                </p>
              </div>
            ))
          )}
          <div ref={bottomRef} />
        </div>
      </div>
    </div>
  );
};

export default TranscriptPanel;

