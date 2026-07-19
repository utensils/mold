#!/usr/bin/env ruby
# frozen_string_literal: true

require "base64"
require "json"
require "net/http"
require "openssl"
require "uri"

bundle_id, version, build_number = ARGV
abort "usage: wait-for-testflight.rb BUNDLE_ID VERSION BUILD" unless build_number

def b64(value)
  Base64.urlsafe_encode64(value, padding: false)
end

def token
  now = Time.now.to_i
  header = b64({ alg: "ES256", kid: ENV.fetch("ASC_KEY_ID"), typ: "JWT" }.to_json)
  payload = b64({ iss: ENV.fetch("ASC_ISSUER_ID"), iat: now, exp: now + 1_100, aud: "appstoreconnect-v1" }.to_json)
  digest = OpenSSL::Digest::SHA256.digest("#{header}.#{payload}")
  signature = OpenSSL::PKey::EC.new(ENV.fetch("PRIVATE_KEY")).dsa_sign_asn1(digest)
  r, s = OpenSSL::ASN1.decode(signature).value.map(&:value)
  raw = [r.to_s(16).rjust(64, "0"), s.to_s(16).rjust(64, "0")].pack("H*H*")
  "#{header}.#{payload}.#{b64(raw)}"
end

def get(path)
  uri = URI("https://api.appstoreconnect.apple.com#{path}")
  request = Net::HTTP::Get.new(uri)
  request["Authorization"] = "Bearer #{token}"
  response = Net::HTTP.start(uri.host, uri.port, use_ssl: true, open_timeout: 15, read_timeout: 60) do |http|
    http.request(request)
  end
  raise "App Store Connect #{response.code}: #{response.body}" unless response.is_a?(Net::HTTPSuccess)
  JSON.parse(response.body)
end

app = get("/v1/apps?filter[bundleId]=#{URI.encode_www_form_component(bundle_id)}").fetch("data").first
abort "No App Store Connect app for #{bundle_id}" unless app
deadline = Time.now + 45 * 60
loop do
  path = "/v1/builds?filter[app]=#{app.fetch('id')}&filter[version]=#{URI.encode_www_form_component(build_number)}&include=preReleaseVersion&limit=10"
  build = get(path).fetch("data").find { |row| row.dig("attributes", "version") == build_number }
  state = build&.dig("attributes", "processingState")
  puts "TestFlight #{version} (#{build_number}): #{state || 'not visible'}"
  exit 0 if state == "VALID"
  abort "TestFlight processing ended in #{state}" if %w[FAILED INVALID].include?(state)
  abort "Timed out waiting for TestFlight" if Time.now >= deadline
  sleep 30
end
