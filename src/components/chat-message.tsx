import { cn } from '@/lib/utils';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import {
  Bot,
  FileText,
  User,
  ChevronDown,
  Wand2,
} from 'lucide-react';
import { Skeleton } from './ui/skeleton';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from './ui/collapsible';
import { Button } from './ui/button';
import { GolemStats } from './golem-stats';
import React from 'react';
import type { Message } from '@/app/page';

type ChatMessageProps = {
  message: Message;
};

const AssistantMessageContent = ({
  content,
  cosmicThoughts,
  golemStats,
}: {
  content: string;
  cosmicThoughts?: string;
  golemStats?: any;
}) => {
  return (
    <div className="flex flex-col gap-2">
      {cosmicThoughts && (
        <Collapsible>
          <CollapsibleTrigger asChild>
            <button className="flex w-full items-center gap-2 rounded-md p-2 text-xs text-muted-foreground hover:bg-card-foreground/5 hover:text-foreground -m-2 mb-0">
              <Wand2 className="h-4 w-4" />
              <span>Cosmic Thoughts</span>
              <ChevronDown className="ml-auto h-4 w-4 transition-transform data-[state=open]:rotate-180" />
            </button>
          </CollapsibleTrigger>
          <CollapsibleContent className="prose prose-sm prose-invert mt-2 border-t border-card-foreground/10 pt-2 text-muted-foreground">
            <p className="whitespace-pre-wrap leading-relaxed">{cosmicThoughts}</p>
          </CollapsibleContent>
        </Collapsible>
      )}

      <p className="whitespace-pre-wrap leading-relaxed">{content}</p>

      {golemStats && (
        <div className="mt-2 border-t border-primary-foreground/20 pt-2 text-xs text-card-foreground/80">
          <Collapsible>
            <CollapsibleTrigger asChild>
              <Button
                variant="ghost"
                size="sm"
                className="w-full justify-start gap-2 p-0 h-auto text-card-foreground/80 hover:bg-transparent hover:text-card-foreground"
              >
                <ChevronDown className="h-4 w-4 transition-transform data-[state=open]:rotate-180" />
                Show Golem Stats
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <GolemStats stats={golemStats} />
            </CollapsibleContent>
          </Collapsible>
        </div>
      )}
    </div>
  );
};

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';
  return (
    <div
      className={cn(
        'flex items-start gap-3',
        isUser ? 'justify-end' : 'justify-start'
      )}
    >
      {!isUser && (
        <Avatar className="h-8 w-8">
          <AvatarFallback>
            <Bot className="h-5 w-5 text-primary" />
          </AvatarFallback>
        </Avatar>
      )}
      <div
        className={cn(
          'max-w-2xl rounded-lg p-3 text-sm shadow-md',
          isUser
            ? 'rounded-br-none bg-primary text-primary-foreground'
            : 'rounded-bl-none bg-card text-card-foreground'
        )}
      >
        {message.file && (
          <div className="mb-2 flex items-center gap-2 rounded-md border border-primary-foreground/20 bg-primary-foreground/10 p-2">
            <FileText className="h-4 w-4 shrink-0" />
            <span className="truncate font-medium">{message.file.name}</span>
          </div>
        )}
        
        {isUser ? (
          <p className="whitespace-pre-wrap leading-relaxed">
            {message.content}
          </p>
        ) : (
          <AssistantMessageContent
            content={message.content}
            cosmicThoughts={message.cosmicThoughts}
            golemStats={message.golemStats}
          />
        )}
      </div>
      {isUser && (
        <Avatar className="h-8 w-8">
          <AvatarFallback>
            <User className="h-5 w-5 text-primary" />
          </AvatarFallback>
        </Avatar>
      )}
    </div>
  );
}

export function LoadingMessage() {
  return (
    <div className="flex items-start justify-start gap-3">
      <Avatar className="h-8 w-8">
        <AvatarFallback>
          <Bot className="h-5 w-5 text-primary" />
        </AvatarFallback>
      </Avatar>
      <div className="max-w-md rounded-lg rounded-bl-none bg-card p-3 text-card-foreground shadow-md">
        <div className="flex items-center gap-2">
          <Skeleton className="h-2 w-2 animate-bounce rounded-full [animation-delay:-0.3s]" />
          <Skeleton className="h-2 w-2 animate-bounce rounded-full [animation-delay:-0.15s]" />
          <Skeleton className="h-2 w-2 animate-bounce rounded-full" />
        </div>
      </div>
    </div>
  );
}
