import { cn } from '@/lib/utils';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import {
  Bot,
  FileText,
  User,
  ChevronDown,
  Lightbulb,
  MessageCircle,
  Atom,
} from 'lucide-react';
import { Skeleton } from './ui/skeleton';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from './ui/collapsible';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { Button } from './ui/button';
import { GolemStats } from './golem-stats';
import React from 'react';

export type Message = {
  role: 'user' | 'assistant';
  content: string;
  file?: {
    name: string;
  };
  golemStats?: any;
};

type ChatMessageProps = {
  message: Message;
};

const AssistantMessageContent = ({
  content,
  golemStats,
}: {
  content: string;
  golemStats?: any;
}) => {
  const sections = React.useMemo(() => {
    const parts = content.split(/###\s+/).filter((part) => part.trim() !== '');
    const result = {
      recommendation: '',
      directResponse: '',
      aetherAnalysis: '',
    };
    let parsed = false;

    parts.forEach((part) => {
      if (part.startsWith('Golem Recommendation')) {
        result.recommendation = part.replace('Golem Recommendation', '').trim();
        parsed = true;
      } else if (part.startsWith('Direct Response')) {
        result.directResponse = part.replace('Direct Response', '').trim();
        parsed = true;
      } else if (part.startsWith('Aether Analysis')) {
        result.aetherAnalysis = part.replace('Aether Analysis', '').trim();
        parsed = true;
      }
    });

    return parsed ? result : null;
  }, [content]);

  return (
    <>
      {sections ? (
        <Accordion
          type="multiple"
          className="w-full -mx-3 text-sm"
          defaultValue={['direct-response']}
        >
          {sections.directResponse && (
            <AccordionItem value="direct-response" className="border-b-0 px-3">
              <AccordionTrigger className="py-2 font-semibold hover:no-underline">
                <div className="flex items-center gap-2">
                  <MessageCircle className="h-4 w-4 text-blue-500" />
                  Direct Response
                </div>
              </AccordionTrigger>
              <AccordionContent className="pt-1 pb-2 whitespace-pre-wrap leading-relaxed">
                {sections.directResponse}
              </AccordionContent>
            </AccordionItem>
          )}
          {sections.recommendation && (
            <AccordionItem value="recommendation" className="border-b-0 px-3">
              <AccordionTrigger className="py-2 font-semibold hover:no-underline">
                <div className="flex items-center gap-2">
                  <Lightbulb className="h-4 w-4 text-yellow-500" />
                  Golem Recommendation
                </div>
              </AccordionTrigger>
              <AccordionContent className="pt-1 pb-2 whitespace-pre-wrap leading-relaxed">
                {sections.recommendation}
              </AccordionContent>
            </AccordionItem>
          )}
          {sections.aetherAnalysis && (
            <AccordionItem value="aether-analysis" className="border-b-0 px-3">
              <AccordionTrigger className="py-2 font-semibold hover:no-underline">
                <div className="flex items-center gap-2">
                  <Atom className="h-4 w-4 text-purple-500" />
                  Aether Analysis
                </div>
              </AccordionTrigger>
              <AccordionContent className="pt-1 pb-2 whitespace-pre-wrap leading-relaxed">
                {sections.aetherAnalysis}
              </AccordionContent>
            </AccordionItem>
          )}
        </Accordion>
      ) : (
        <p className="whitespace-pre-wrap leading-relaxed">{content}</p>
      )}

      {golemStats && (
        <div className="mt-2 border-t pt-2 text-xs text-card-foreground/80">
          <Collapsible>
            <CollapsibleTrigger asChild>
              <Button
                variant="ghost"
                size="sm"
                className="w-full justify-start gap-2 p-0 h-auto text-card-foreground/80 hover:bg-transparent hover:text-card-foreground"
              >
                <ChevronDown className="h-4 w-4 transition-transform group-data-[state=open]:rotate-180" />
                Show Golem Stats
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <GolemStats stats={golemStats} />
            </CollapsibleContent>
          </Collapsible>
        </div>
      )}
    </>
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
        {message.content &&
          (isUser ? (
            <p className="whitespace-pre-wrap leading-relaxed">
              {message.content}
            </p>
          ) : (
            <AssistantMessageContent
              content={message.content}
              golemStats={message.golemStats}
            />
          ))}
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
