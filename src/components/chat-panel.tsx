"use client";

import { ollamaChat } from '@/ai/flows/ollama-chat';
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
import { useToast } from '@/hooks/use-toast';
import { Bot, SendHorizonal, Settings2 } from 'lucide-react';
import React, { useEffect, useRef, useState } from 'react';
import { ChatMessage, LoadingMessage, type Message } from './chat-message';
import { Label } from './ui/label';

export function ChatPanel() {
  const { toast } = useToast();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [temperature, setTemperature] = useState(0.7);
  const [context, setContext] = useState<string[]>([]);
  
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTo({
        top: scrollAreaRef.current.scrollHeight,
        behavior: 'smooth',
      });
    }
  }, [messages, isLoading]);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const result = await ollamaChat({
        prompt: input,
        temperature,
        context,
      });

      if (result.response) {
        const assistantMessage: Message = {
          role: 'assistant',
          content: result.response,
        };
        setMessages((prev) => [...prev, assistantMessage]);
        setContext(result.updatedContext || []);
      }
    } catch (error) {
      console.error('Error calling ollamaChat:', error);
      toast({
        variant: 'destructive',
        title: 'Error',
        description:
          'Failed to get a response from the AI. Please try again.',
      });
      // remove the user message that failed
      setMessages((prev) => prev.slice(0, prev.length - 1));
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as any);
    }
  };

  return (
    <Card className="w-full max-w-2xl h-[90vh] flex flex-col shadow-2xl">
      <CardHeader className="flex flex-row items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-primary/10 rounded-full">
            <Bot className="w-6 h-6 text-primary" />
          </div>
          <CardTitle className="font-headline text-2xl">QwenChat</CardTitle>
        </div>
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
      </CardHeader>
      <Separator />
      <CardContent className="flex-1 p-0 overflow-hidden">
        <ScrollArea className="h-full" ref={scrollAreaRef}>
          <div className="p-6 space-y-6">
            {messages.length === 0 && !isLoading && (
              <div className="flex flex-col items-center justify-center h-full text-center text-muted-foreground">
                <Bot className="w-12 h-12 mb-4" />
                <p className="text-lg">Start a conversation with Qwen!</p>
                <p className="text-sm">Ask me anything, or try an example.</p>
              </div>
            )}
            {messages.map((message, index) => (
              <ChatMessage key={index} message={message} />
            ))}
            {isLoading && <LoadingMessage />}
          </div>
        </ScrollArea>
      </CardContent>
      <CardFooter className="p-4 border-t">
        <form onSubmit={handleSubmit} className="flex w-full items-end gap-2">
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your message..."
            className="flex-1 resize-none min-h-[40px] max-h-40 bg-card"
            rows={1}
            disabled={isLoading}
          />
          <Button
            type="submit"
            size="icon"
            disabled={isLoading || !input.trim()}
            className="bg-accent hover:bg-accent/90 shrink-0"
            aria-label="Send message"
          >
            <SendHorizonal className="h-5 w-5" />
          </Button>
        </form>
      </CardFooter>
    </Card>
  );
}
