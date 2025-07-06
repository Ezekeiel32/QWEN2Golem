"use client";

import { Button } from '@/components/ui/button';
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Slider } from '@/components/ui/slider';
import { Textarea } from '@/components/ui/textarea';
import { Bot, Paperclip, SendHorizonal, Settings2, X, MessageSquarePlus } from 'lucide-react';
import React, { useEffect, useRef, useState } from 'react';
import { ChatMessage, LoadingMessage, type Message } from './chat-message';
import { Label } from './ui/label';
import { Badge } from './ui/badge';

type ChatPanelProps = {
  messages: Message[];
  onSendMessage: (input: string, temperature: number, file: File | null) => Promise<void>;
  isLoading: boolean;
  isChatSelected: boolean;
};

export function ChatPanel({
  messages,
  onSendMessage,
  isLoading,
  isChatSelected,
}: ChatPanelProps) {
  const [input, setInput] = useState('');
  const [temperature, setTemperature] = useState(0.7);
  const [file, setFile] = useState<File | null>(null);

  const scrollAreaViewportRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (scrollAreaViewportRef.current) {
      scrollAreaViewportRef.current.scrollTop = scrollAreaViewportRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
    }
    if (e.target) {
      e.target.value = '';
    }
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if ((!input.trim() && !file) || isLoading || !isChatSelected) return;

    await onSendMessage(input, temperature, file);
    setInput('');
    setFile(null);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as any);
    }
  };

  return (
    <Card className="w-full h-full flex flex-col shadow-2xl">
      <CardHeader className="flex flex-row items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-primary/10 rounded-full">
            <Bot className="w-6 h-6 text-primary" />
          </div>
          <CardTitle className="font-headline text-2xl">QwenChat</CardTitle>
        </div>
        { isChatSelected && (
            <Popover>
            <PopoverTrigger asChild>
                <Button variant="ghost" size="icon">
                <Settings2 className="h-5 w-5" />
                <span className="sr-only">Settings</span>
                </Button>
            </PopoverTrigger>
            <PopoverContent className="w-80">
                <div className="grid gap-4">
                <div className="space-y-2">
                    <h4 className="font-medium leading-none">Settings</h4>
                    <p className="text-sm text-muted-foreground">
                    Adjust chatbot parameters.
                    </p>
                </div>
                <Separator />
                <div className="grid gap-2">
                    <Label htmlFor="temperature">Temperature: {temperature.toFixed(1)}</Label>
                    <Slider
                    id="temperature"
                    min={0}
                    max={1}
                    step={0.1}
                    value={[temperature]}
                    onValueChange={(value) => setTemperature(value[0])}
                    />
                    <p className="text-xs text-muted-foreground">
                    Controls randomness. Lower values are more deterministic.
                    </p>
                </div>
                </div>
            </PopoverContent>
            </Popover>
        )}
      </CardHeader>
      <Separator />
      <CardContent className="flex-1 p-0 overflow-hidden">
        <ScrollArea className="h-full" viewportRef={scrollAreaViewportRef}>
          <div className="p-6 space-y-6">
            {!isChatSelected ? (
              <div className="flex flex-col items-center justify-center h-full text-center text-muted-foreground">
                <MessageSquarePlus className="w-12 h-12 mb-4" />
                <p className="text-lg">Start a new conversation</p>
                <p className="text-sm">Click the "New Chat" button in the sidebar to begin.</p>
              </div>
            ) : messages.length === 0 && !isLoading ? (
              <div className="flex flex-col items-center justify-center h-full text-center text-muted-foreground">
                <Bot className="w-12 h-12 mb-4" />
                <p className="text-lg">Start a conversation with Qwen!</p>
                <p className="text-sm">Ask me anything, or attach a file.</p>
              </div>
            ) : (
                <>
                    {messages.map((message, index) => (
                    <ChatMessage key={index} message={message} />
                    ))}
                    {isLoading && <LoadingMessage />}
                </>
            )}
          </div>
        </ScrollArea>
      </CardContent>
      {isChatSelected && (
        <CardFooter className="p-4 border-t flex flex-col items-start gap-2">
          {file && (
            <div className="flex items-center gap-2">
              <Badge variant="secondary">
                <Paperclip className="h-3 w-3 mr-1" />
                {file.name}
              </Badge>
              <Button variant="ghost" size="icon" className="h-6 w-6" onClick={() => setFile(null)}>
                <X className="h-4 w-4" />
              </Button>
            </div>
          )}
          <form onSubmit={handleSubmit} className="flex w-full items-end gap-2">
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              className="hidden"
              accept=".txt,.csv,.py,.js,.ts,.html,.css,.json,.md"
            />
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={() => fileInputRef.current?.click()}
              disabled={isLoading}
              className="shrink-0"
              aria-label="Attach file"
            >
              <Paperclip className="h-5 w-5" />
            </Button>
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type your message or attach a file..."
              className="flex-1 resize-none min-h-[40px] max-h-40 bg-card"
              rows={1}
              disabled={isLoading}
            />
            <Button
              type="submit"
              size="icon"
              disabled={isLoading || (!input.trim() && !file)}
              className="bg-accent hover:bg-accent/90 shrink-0"
              aria-label="Send message"
            >
              <SendHorizonal className="h-5 w-5" />
            </Button>
          </form>
        </CardFooter>
      )}
    </Card>
  );
}
